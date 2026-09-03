"""Find +EV Premier League bets by comparing live bookmaker odds
(via the-odds-api.com) against Dixon-Coles model probabilities from
predict_match.py.

Scans two markets: 1X2 (h2h) and Over/Under total goals. Only the 2.5
goals line is evaluated — that's the line predict_match computes by
default — so other totals lines a book may offer are ignored.

Setup:
    Get a free API key at https://the-odds-api.com and either export it
        export ODDS_API_KEY=your_key_here
    or pass it with --api-key.

Usage:
    python find_ev_bets.py                       # live odds, default thresholds
    python find_ev_bets.py --min-ev 0.05 --bankroll 500
    python find_ev_bets.py --market totals        # only over/under 2.5
    python find_ev_bets.py --dry-run              # sample data, no API call/key needed
"""

import argparse
import os
import sys

import pandas as pd
import requests

import predict_match

SPORT = "soccer_epl"
TOTALS_LINE = 2.5  # the only Over/Under line the model prices, and the one we score

# Odds API market key -> the API-request market names it needs.
MARKET_GROUPS = {"h2h": ["h2h"], "totals": ["totals"], "all": ["h2h", "totals"]}

# The Odds API uses each club's full official name; our model (built from
# Understat data) uses shorter names. Map API name -> model name here.
TEAM_NAME_ALIASES = {
    "Brighton and Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
}


def fetch_live_odds(api_key, sport=SPORT, region="uk", markets=("h2h", "totals")):
    """Fetches upcoming match odds from bookmakers in decimal format."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    res = requests.get(url, params=params, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"Odds API error {res.status_code}: {res.text}")
    return res.json()


def to_model_name(team_name, known_teams):
    if team_name in known_teams:
        return team_name
    return TEAM_NAME_ALIASES.get(team_name, team_name)


def build_model_predictions(live_fixtures, params):
    """Runs predict_match.predict() for each fixture whose teams are known
    to the model, keyed by the odds API's own "Home vs Away" team names so
    it lines up with evaluate_market_opportunities()."""
    known_teams = params["teams"]
    predictions = {}

    for game in live_fixtures:
        api_home, api_away = game["home_team"], game["away_team"]
        model_home = to_model_name(api_home, known_teams)
        model_away = to_model_name(api_away, known_teams)

        if model_home not in known_teams or model_away not in known_teams:
            print(
                f"Skipping {api_home} vs {api_away}: team(s) not in model "
                f"({model_home!r}, {model_away!r})",
                file=sys.stderr,
            )
            continue

        result = predict_match.predict(model_home, model_away, params)
        match_key = f"{api_home} vs {api_away}"
        predictions[match_key] = {
            "Home Win": result["home_win"],
            "Draw": result["draw"],
            "Away Win": result["away_win"],
            f"Over {TOTALS_LINE}": result["goals_over"],
            f"Under {TOTALS_LINE}": result["goals_under"],
        }

    return predictions


def calculate_ev_and_kelly(prob, decimal_odds, bankroll, kelly_frac):
    """Calculates EV and Fractional Kelly stake size."""
    ev = (prob * decimal_odds) - 1.0

    # Kelly Formula: f* = (p*b - q) / b
    b = decimal_odds - 1.0
    q = 1.0 - prob
    full_kelly = (prob * b - q) / b if b > 0 else 0

    stake = max(0, full_kelly * kelly_frac * bankroll) if ev > 0 else 0.0
    return round(ev, 4), round(stake, 2)


def resolve_outcome(market_key, outcome, home_team, away_team, probs):
    """Map one bookmaker outcome to a (display label, model probability) pair,
    or None if the model doesn't price it (unknown selection, or a totals line
    other than TOTALS_LINE)."""
    if market_key == "h2h":
        name = outcome["name"]
        if name == home_team:
            return name, probs.get("Home Win")
        if name == away_team:
            return name, probs.get("Away Win")
        return name, probs.get("Draw")  # the API names the draw "Draw"

    if market_key == "totals":
        if outcome.get("point") != TOTALS_LINE:
            return None
        side = outcome["name"]  # "Over" / "Under"
        label = f"{side} {TOTALS_LINE:g}"
        return label, probs.get(label)

    return None


def evaluate_market_opportunities(
    live_fixtures, model_predictions, bankroll, min_ev, kelly_frac, markets=("h2h", "totals")
):
    """Compares live odds against model probabilities to extract +EV bets."""
    opportunities = []

    for game in live_fixtures:
        home_team = game["home_team"]
        away_team = game["away_team"]
        match_key = f"{home_team} vs {away_team}"

        if match_key not in model_predictions:
            continue

        probs = model_predictions[match_key]

        for bookmaker in game.get("bookmakers", []):
            bookie_name = bookmaker["title"]

            for market in bookmaker.get("markets", []):
                if market["key"] not in markets:
                    continue

                for outcome in market["outcomes"]:
                    resolved = resolve_outcome(market["key"], outcome, home_team, away_team, probs)
                    if resolved is None:
                        continue
                    selection, model_p = resolved
                    if not model_p:
                        continue

                    odds = outcome["price"]
                    ev, stake = calculate_ev_and_kelly(model_p, odds, bankroll, kelly_frac)
                    if ev >= min_ev:
                        opportunities.append(
                            {
                                "Match": match_key,
                                "Market": market["key"],
                                "Bookmaker": bookie_name,
                                "Selection": selection,
                                "Odds": odds,
                                "Model_Prob": f"{model_p:.1%}",
                                "Implied_Prob": f"{(1 / odds):.1%}",
                                "EV_%": f"{ev * 100:+.2f}%",
                                "Rec_Stake": f"${stake}",
                                "_ev": ev,  # numeric key for sorting; dropped before display
                            }
                        )

    df_ev = pd.DataFrame(opportunities)
    if df_ev.empty:
        return df_ev
    return df_ev.sort_values(by="_ev", ascending=False).drop(columns="_ev")


def sample_fixtures():
    """Sample odds payload for --dry-run, so the pipeline is testable without an API key."""
    return [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "title": "Unibet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.95},
                                {"name": "Chelsea", "price": 4.20},
                                {"name": "Draw", "price": 3.60},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.80, "point": 2.5},
                                {"name": "Under", "price": 2.00, "point": 2.5},
                                {"name": "Over", "price": 3.40, "point": 3.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Find +EV Premier League 1X2 and Over/Under 2.5 bets")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"))
    parser.add_argument("--region", default="uk", choices=["uk", "eu", "us", "au"])
    parser.add_argument("--bankroll", type=float, default=1000.00)
    parser.add_argument("--min-ev", type=float, default=0.03, help="Minimum EV threshold, e.g. 0.03 = 3%%")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument(
        "--market",
        default="all",
        choices=list(MARKET_GROUPS),
        help="Which market(s) to scan: h2h (1X2), totals (Over/Under 2.5), or all (default)",
    )
    parser.add_argument("--params", default=predict_match.PARAMS_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Use sample odds instead of calling the live API")
    args = parser.parse_args()

    markets = MARKET_GROUPS[args.market]

    if args.dry_run:
        live_fixtures = sample_fixtures()
    else:
        if not args.api_key:
            raise SystemExit(
                "No API key found. Get a free key from https://the-odds-api.com, then "
                "export ODDS_API_KEY=<key> or pass --api-key."
            )
        live_fixtures = fetch_live_odds(args.api_key, region=args.region, markets=markets)

    params = predict_match.load_params(args.params)
    model_predictions = build_model_predictions(live_fixtures, params)

    results = evaluate_market_opportunities(
        live_fixtures, model_predictions, args.bankroll, args.min_ev, args.kelly_fraction, markets
    )

    if results.empty:
        print(f"No +EV opportunities found (threshold {args.min_ev:.1%}).")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
