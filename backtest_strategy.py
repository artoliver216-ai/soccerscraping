"""backtest_strategy.py — simulate the Dixon-Coles +EV betting system over
the 2025/26 Premier League season and report its bottom line.

Market prices come from football-data.co.uk: the closing 1X2 odds, using
the "MaxC" columns (best price available at kick-off across the tracked
books) — the fair analogue of `find_ev_bets.py` shopping every bookmaker.
The file is fetched once to `odds_2526.csv`.

Method — walk-forward, no lookahead:
  - Matches are processed in date order.
  - Before each match the Dixon-Coles model is (re)fit on ONLY the
    2025/26 matches played before it (`fbref_xg.csv`), refit every
    `--refit-every` matches; the `--min-matches` guard applies.
  - Model 1X2 probabilities are compared to the closing odds. Each
    closing price is first discounted for `--commission` (default 2%,
    treating it as an exchange back price):
        net = 1 + (odds - 1) * (1 - commission)
  - A bet is placed on every outcome with
        EV = p_model * net_odds - 1  >=  --min-ev   (default 0.03)
    staked flat (1 unit) or by fractional Kelly (`--stake kelly`).
  - Bets settle on the actual result.

The only training data is 2025/26 itself, so betting can't start until
the model has enough of the season to fit — the first `--min-train`
matches are training-only. This therefore backtests roughly the back
60% of 2025/26.

Outputs: total ROI %, win rate, profit curve (-> `strategy_curve.csv`
plus an ASCII sketch), and max drawdown.

    python backtest_strategy.py
    python backtest_strategy.py --min-ev 0.05 --commission 0.02 --stake kelly
"""

import argparse
import os

import numpy as np
import pandas as pd
import requests

import backtest as bt
import fit_dixon_coles as fdc
import predict_match as pm
from find_ev_bets import adjust_exchange_odds

ODDS_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
ODDS_CSV = "odds_2526.csv"

# football-data.co.uk team names -> our model's (Understat-derived) names.
FD_TO_MODEL = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
}

# football-data FTR code -> the key predict_match returns.
RESULT_TO_OUTCOME = {"H": "home_win", "D": "draw", "A": "away_win"}
OUTCOME_LABEL = {"H": "Home", "D": "Draw", "A": "Away"}


def load_market(path=ODDS_CSV):
    if not os.path.exists(path):
        print(f"Fetching closing odds -> {path}")
        res = requests.get(ODDS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        res.raise_for_status()
        with open(path, "wb") as f:
            f.write(res.content)

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    for col in ("HomeTeam", "AwayTeam"):
        df[col] = df[col].map(lambda t: FD_TO_MODEL.get(t, t))
    df = df.rename(columns={"MaxCH": "odds_H", "MaxCD": "odds_D", "MaxCA": "odds_A"})
    keep = ["Date", "HomeTeam", "AwayTeam", "odds_H", "odds_D", "odds_A", "FTR"]
    return df.dropna(subset=["odds_H", "odds_D", "odds_A", "FTR"])[keep].sort_values(
        "Date"
    ).reset_index(drop=True)


def kelly_fraction(p, net_odds):
    b = net_odds - 1.0
    q = 1.0 - p
    return max(0.0, (p * b - q) / b) if b > 0 else 0.0


def run_strategy(market, train_all, args):
    bets = []
    params = None
    matches_since_fit = 0
    bankroll = args.bankroll

    for _, row in market.iterrows():
        train = train_all[train_all["Date"] < row["Date"]]
        if len(train) < args.min_train:
            continue

        if params is None or matches_since_fit >= args.refit_every:
            train_f, _ = fdc.filter_sparse_teams(train, args.min_matches, verbose=False)
            teams, attack, defense, gamma, rho, _ = fdc.fit(train_f)
            params = bt.params_from_fit(teams, attack, defense, gamma, rho)
            matches_since_fit = 0
        matches_since_fit += 1

        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in params["teams"] or away not in params["teams"]:
            continue

        pred = pm.predict(home, away, params)

        for code, odds_col in (("H", "odds_H"), ("D", "odds_D"), ("A", "odds_A")):
            p = pred[RESULT_TO_OUTCOME[code]]
            net = adjust_exchange_odds(row[odds_col], args.commission)
            ev = p * net - 1.0
            if ev < args.min_ev:
                continue

            if args.stake == "kelly":
                stake = args.kelly_fraction * kelly_fraction(p, net) * bankroll
            else:
                stake = 1.0
            if stake <= 0:
                continue

            won = row["FTR"] == code
            profit = stake * (net - 1.0) if won else -stake
            bankroll += profit

            bets.append(
                {
                    "Date": row["Date"].date(),
                    "Match": f"{home} vs {away}",
                    "Selection": OUTCOME_LABEL[code],
                    "Odds": round(row[odds_col], 2),
                    "NetOdds": round(net, 3),
                    "ModelProb": round(p, 3),
                    "EV": round(ev, 3),
                    "Stake": round(stake, 3),
                    "Won": won,
                    "Profit": round(profit, 3),
                    "Bankroll": round(bankroll, 2),
                }
            )

    return pd.DataFrame(bets)


def ascii_curve(cum, width=64, height=14):
    """Rough profit-curve sketch: cum is the cumulative-profit series."""
    if len(cum) < 2:
        return "(not enough bets to plot)"
    xs = np.linspace(0, len(cum) - 1, min(width, len(cum))).astype(int)
    sample = cum.iloc[xs].to_numpy()
    lo, hi = sample.min(), sample.max()
    span = hi - lo or 1.0
    rows = []
    for r in range(height, -1, -1):
        level = lo + span * r / height
        line = "".join("█" if v >= level else " " for v in sample)
        marker = ""
        if r == height:
            marker = f" {hi:+.1f}"
        elif r == 0:
            marker = f" {lo:+.1f}"
        rows.append(line + marker)
    zero_row = int(round((0 - lo) / span * height))
    if 0 <= zero_row <= height:
        rows[height - zero_row] = rows[height - zero_row].rstrip() + "  <- break-even"
    return "\n".join(rows)


def summarize(bets, args):
    n = len(bets)
    staked = bets["Stake"].sum()
    profit = bets["Profit"].sum()
    wins = int(bets["Won"].sum())

    cum = bets["Profit"].cumsum()
    equity = args.bankroll + cum
    peak = equity.cummax()
    drawdown = peak - equity
    max_dd = drawdown.max()
    max_dd_pct = (drawdown / peak).max() * 100

    print(f"\n{'='*58}")
    print(f"  STRATEGY BACKTEST — 2025/26  ({args.stake} staking)")
    print(f"{'='*58}")
    print(f"  Betting window     {bets['Date'].iloc[0]}  ->  {bets['Date'].iloc[-1]}")
    print(f"  Bets placed        {n}")
    print(f"  Total staked       {staked:.2f} u")
    print(f"  Total profit       {profit:+.2f} u")
    print(f"  ROI                {profit / staked * 100:+.2f} %")
    print(f"  Win rate           {wins}/{n}  ({wins / n * 100:.1f} %)")
    print(f"  Avg net odds       {bets['NetOdds'].mean():.2f}")
    print(f"  Avg model edge     {bets['EV'].mean() * 100:+.1f} %")
    print(f"  Final bankroll     {equity.iloc[-1]:.2f} u  (start {args.bankroll:.0f})")
    print(f"  Max drawdown       {max_dd:.2f} u  ({max_dd_pct:.1f} % of peak)")
    print(f"{'='*58}")

    by_sel = bets.groupby("Selection").agg(
        bets=("Profit", "size"), profit=("Profit", "sum"), win_rate=("Won", "mean")
    )
    by_sel["win_rate"] = (by_sel["win_rate"] * 100).round(1)
    by_sel["profit"] = by_sel["profit"].round(2)
    print("\nBy selection:")
    print(by_sel.to_string())

    print("\nProfit curve (cumulative units, chronological):")
    print(ascii_curve(cum))


def main():
    parser = argparse.ArgumentParser(description="Backtest the Dixon-Coles +EV betting system over 2025/26")
    parser.add_argument("--xg-csv", default=fdc.CSV_PATH)
    parser.add_argument("--odds-csv", default=ODDS_CSV)
    parser.add_argument("--min-train", type=int, default=150, help="2025/26 matches to accumulate before betting starts")
    parser.add_argument("--refit-every", type=int, default=10, help="Rolling refit cadence, in matches")
    parser.add_argument("--min-matches", type=int, default=fdc.MIN_MATCHES)
    parser.add_argument("--min-ev", type=float, default=0.03, help="EV threshold to place a bet (0.03 = +3%%)")
    parser.add_argument("--commission", type=float, default=0.02, help="Exchange commission on winnings")
    parser.add_argument("--stake", choices=["flat", "kelly"], default="flat")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--bankroll", type=float, default=100.0, help="Starting bankroll in units (for the curve / drawdown)")
    parser.add_argument("--curve-out", default="strategy_curve.csv")
    args = parser.parse_args()

    market = load_market(args.odds_csv)
    train_all = fdc.load_matches(args.xg_csv)
    train_all = train_all[train_all["Date"] <= market["Date"].max()]  # 2025/26 only

    bets = run_strategy(market, train_all, args)
    if bets.empty:
        raise SystemExit("No qualifying bets — try a lower --min-ev or --min-train.")

    bets["CumProfit"] = bets["Profit"].cumsum().round(3)
    bets.to_csv(args.curve_out, index=False)
    summarize(bets, args)
    print(f"\nPer-bet ledger written to {args.curve_out}")


if __name__ == "__main__":
    main()
