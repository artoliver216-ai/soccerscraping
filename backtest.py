"""Walk-forward backtest of the Dixon-Coles model's calibration.

Matches are processed in date order. For each test match, the model is
refit on *only* the matches that kicked off before it, then its
predictions are scored against the actual result, in two markets:

  - 1X2: Home / Draw / Away  (log-loss, ranked probability score, Brier,
    argmax accuracy)
  - Over/Under 2.5 total goals  (binary log-loss, Brier, accuracy)

Each metric is shown next to a base-rate baseline (the running frequency
of that outcome in the training set), so a number has something to be
"good" or "bad" relative to. Both markets also get a calibration table.

Refitting for every single match is slow, so by default the model is
refit every `--refit-every` matches and reused in between; pass
`--refit-every 1` for a match-by-match backtest.

There is no betting-ROI simulation here: the odds API's free tier only
serves current/upcoming fixtures, so there are no historical closing
lines to bet against.

    python backtest.py
    python backtest.py --min-train 200 --refit-every 1 --csv-out bt.csv
"""

import argparse

import numpy as np
import pandas as pd

import fit_dixon_coles as fdc
import predict_match as pm

# Ordered worst-to-best for the home side, which RPS relies on.
OUTCOMES = ["home_win", "draw", "away_win"]
_OUTCOME_IDX = {o: i for i, o in enumerate(OUTCOMES)}

TOTALS_LINE = 2.5  # the Over/Under line predict_match prices by default


def parse_score(score):
    home_goals, away_goals = (int(x) for x in str(score).split("-"))
    return home_goals, away_goals


def actual_outcome(score):
    home_goals, away_goals = parse_score(score)
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def params_from_fit(teams, attack, defense, gamma, rho):
    """Assemble the dict shape predict_match.predict() expects from a fit."""
    return {
        "teams": {
            t: {"attack": float(a), "defense": float(b)}
            for t, a, b in zip(teams, attack, defense)
        },
        "home_advantage": float(gamma),
        "rho": float(rho),
    }


def walk_forward(df, min_train, refit_every, min_matches):
    df = df.sort_values("Date").reset_index(drop=True)
    df["_outcome"] = df["Score"].map(actual_outcome)
    df["_over25"] = df["Score"].map(lambda s: int(sum(parse_score(s)) > TOTALS_LINE))

    records = []
    skipped = 0
    params = None
    base_rates = None
    base_over = None
    last_fit_idx = None

    for i in range(min_train, len(df)):
        if params is None or i - last_fit_idx >= refit_every:
            train = df.iloc[:i]
            train_f, _ = fdc.filter_sparse_teams(train, min_matches, verbose=False)
            teams, attack, defense, gamma, rho, _ = fdc.fit(train_f)
            params = params_from_fit(teams, attack, defense, gamma, rho)
            base_rates = (
                train["_outcome"].value_counts(normalize=True).reindex(OUTCOMES).fillna(0.0)
            )
            base_over = train["_over25"].mean()
            last_fit_idx = i

        row = df.iloc[i]
        if row["Home"] not in params["teams"] or row["Away"] not in params["teams"]:
            skipped += 1
            continue

        pred = pm.predict(row["Home"], row["Away"], params)
        records.append(
            {
                "Date": row["Date"].date(),
                "Home": row["Home"],
                "Away": row["Away"],
                "actual": row["_outcome"],
                "over25": row["_over25"],
                "p_home": pred["home_win"],
                "p_draw": pred["draw"],
                "p_away": pred["away_win"],
                "p_over": pred["goals_over"],
                "b_home": base_rates["home_win"],
                "b_draw": base_rates["draw"],
                "b_away": base_rates["away_win"],
                "b_over": base_over,
            }
        )

    return pd.DataFrame(records), skipped


# --- 1X2 (multiclass) scoring -------------------------------------------------

def _probs_and_onehot(results, cols):
    probs = results[cols].to_numpy(dtype=float)
    y = results["actual"].map(_OUTCOME_IDX).to_numpy()
    onehot = np.eye(3)[y]
    return probs, onehot, y


def score_1x2(results, cols):
    """log-loss, RPS, Brier and accuracy for the probability columns `cols`
    (given in OUTCOMES order)."""
    probs, onehot, y = _probs_and_onehot(results, cols)

    p_actual = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    log_loss = -np.mean(np.log(p_actual))
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))

    # RPS: mean squared error between the two CDFs over ordered outcomes,
    # divided by (categories - 1). With 3 categories only the first two
    # cumulative terms are non-trivial.
    cdf_pred = np.cumsum(probs, axis=1)[:, :2]
    cdf_obs = np.cumsum(onehot, axis=1)[:, :2]
    rps = np.mean(np.sum((cdf_pred - cdf_obs) ** 2, axis=1) / 2)

    accuracy = np.mean(np.argmax(probs, axis=1) == y)
    return {"log_loss": log_loss, "rps": rps, "brier": brier, "accuracy": accuracy}


def calibration_1x2(results, cols, bins=10):
    """Pool all predicted probabilities (across the 3 outcomes), bin them, and
    compare mean predicted probability to observed frequency in each bin."""
    probs, onehot, _ = _probs_and_onehot(results, cols)
    return _calibration(probs.reshape(-1), onehot.reshape(-1), bins)


# --- Over/Under (binary) scoring ---------------------------------------------

def score_binary(p, y):
    """Binary log-loss, Brier and accuracy for P(event) `p` vs indicator `y`."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    accuracy = np.mean((p > 0.5) == (y > 0.5))
    return {"log_loss": log_loss, "brier": brier, "accuracy": accuracy}


def calibration_binary(p, y, bins=10):
    return _calibration(np.asarray(p, dtype=float), np.asarray(y, dtype=float), bins)


def _calibration(p_flat, e_flat, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p_flat, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        mask = idx == k
        if not mask.any():
            continue
        rows.append(
            {
                "prob_bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}",
                "n": int(mask.sum()),
                "mean_pred": p_flat[mask].mean(),
                "observed": e_flat[mask].mean(),
            }
        )
    return pd.DataFrame(rows)


# --- reporting --------------------------------------------------------------

def _metric_table(metric_names, model, baseline):
    tbl = pd.DataFrame(
        {
            "metric": metric_names,
            "model": [model[m] for m in metric_names],
            "base_rate": [baseline[m] for m in metric_names],
        }
    )
    tbl["delta"] = tbl["model"] - tbl["base_rate"]
    return tbl


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the Dixon-Coles model")
    parser.add_argument("--csv", default=fdc.CSV_PATH)
    parser.add_argument("--min-train", type=int, default=150, help="Matches to train on before the first prediction")
    parser.add_argument("--refit-every", type=int, default=10, help="Refit cadence, in matches (1 = every match)")
    parser.add_argument("--min-matches", type=int, default=fdc.MIN_MATCHES, help="Passed through to the sparse-team filter")
    parser.add_argument("--bins", type=int, default=10, help="Calibration-table bin count")
    parser.add_argument("--csv-out", help="Optional path to write per-match predictions")
    args = parser.parse_args()

    df = fdc.load_matches(args.csv)
    if args.min_train >= len(df):
        raise SystemExit(f"--min-train ({args.min_train}) must be less than the {len(df)} matches available.")

    results, skipped = walk_forward(df, args.min_train, args.refit_every, args.min_matches)
    if results.empty:
        raise SystemExit("No matches could be scored (every test fixture involved a filtered-out team).")

    print(f"Tested {len(results)} matches ({skipped} skipped: team not in the fit at prediction time)")

    # 1X2
    model = score_1x2(results, ["p_home", "p_draw", "p_away"])
    baseline = score_1x2(results, ["b_home", "b_draw", "b_away"])
    print("\n1X2 (Home / Draw / Away):")
    print(
        _metric_table(["log_loss", "rps", "brier", "accuracy"], model, baseline).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print(f"\nCalibration ({args.bins} bins, all outcomes pooled):")
    print(
        calibration_1x2(results, ["p_home", "p_draw", "p_away"], bins=args.bins).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )

    # Over/Under 2.5
    ou_model = score_binary(results["p_over"], results["over25"])
    ou_base = score_binary(results["b_over"], results["over25"])
    print(f"\nOver/Under {TOTALS_LINE} goals (scoring P(Over)):")
    print(
        _metric_table(["log_loss", "brier", "accuracy"], ou_model, ou_base).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print(f"\nCalibration of P(Over {TOTALS_LINE}) ({args.bins} bins):")
    print(
        calibration_binary(results["p_over"], results["over25"], bins=args.bins).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )

    print("\n  (lower is better for log_loss / rps / brier; higher for accuracy)")

    if args.csv_out:
        results.to_csv(args.csv_out, index=False)
        print(f"\nWrote per-match predictions to {args.csv_out}")


if __name__ == "__main__":
    main()
