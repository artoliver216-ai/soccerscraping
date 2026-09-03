"""Find +EV Premier League 1X2 bets by comparing live bookmaker odds
(via the-odds-api.com) against Dixon-Coles model probabilities from
predict_match.py.

Setup:
    Get a free API key at https://the-odds-api.com and either export it
        export ODDS_API_KEY=your_key_here
    or pass it with --api-key.

Usage:
    python find_ev_bets.py                       # live odds, default thresholds
    python find_ev_bets.py --min-ev 0.05 --bankroll 500
    python find_ev_bets.py --dry-run              # sample data, no API call/key needed
"""

import argparse
import os
import sys

import pandas as pd
import requests

import predict_match

SPORT = "soccer_epl"

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


def fetch_live_odds(api_key, sport=SPORT, region="uk"):
    """Fetches upcoming match odds from bookmakers in decimal format."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {"apiKey": api_key, "regions": region, "markets": "h2h", "oddsFormat": "decimal"}
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


def evaluate_market_opportunities(live_fixtures, model_predictions, bankroll, min_ev, kelly_frac):
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
                if market["key"] != "h2h":
                    continue

                for outcome in market["outcomes"]:
                    selection = outcome["name"]
                    odds = outcome["price"]

                    if selection == home_team:
                        model_p = probs.get("Home Win")
                    elif selection == away_team:
                        model_p = probs.get("Away Win")
                    else:
                        model_p = probs.get("Draw")

                    if not model_p:
                        continue

                    ev, stake = calculate_ev_and_kelly(model_p, odds, bankroll, kelly_frac)
                    if ev >= min_ev:
                        opportunities.append(
                            {
                                "Match": match_key,
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
                        }
                    ],
                }
            ],
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Find +EV Premier League 1X2 bets")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"))
    parser.add_argument("--region", default="uk", choices=["uk", "eu", "us", "au"])
    parser.add_argument("--bankroll", type=float, default=1000.00)
    parser.add_argument("--min-ev", type=float, default=0.03, help="Minimum EV threshold, e.g. 0.03 = 3%%")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--params", default=predict_match.PARAMS_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Use sample odds instead of calling the live API")
    args = parser.parse_args()

    if args.dry_run:
        live_fixtures = sample_fixtures()
    else:
        if not args.api_key:
            raise SystemExit(
                "No API key found. Get a free key from https://the-odds-api.com, then "
                "export ODDS_API_KEY=<key> or pass --api-key."
            )
        live_fixtures = fetch_live_odds(args.api_key, region=args.region)

    params = predict_match.load_params(args.params)
    model_predictions = build_model_predictions(live_fixtures, params)

    results = evaluate_market_opportunities(
        live_fixtures, model_predictions, args.bankroll, args.min_ev, args.kelly_fraction
    )

    if results.empty:
        print(f"No +EV opportunities found (threshold {args.min_ev:.1%}).")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
