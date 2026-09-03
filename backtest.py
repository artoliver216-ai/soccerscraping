"""Walk-forward backtest of the Dixon-Coles model's 1X2 calibration.

Matches are processed in date order. For each test match, the model is
refit on *only* the matches that kicked off before it, then its predicted
Home / Draw / Away probabilities are scored against the actual result.
Reports log-loss, ranked probability score (RPS), multiclass Brier,
argmax accuracy, and a calibration table — each alongside a base-rate
baseline (the running Home/Draw/Away frequency in the training set), so a
number has something to be "good" or "bad" relative to.

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


def actual_outcome(score):
    home_goals, away_goals = (int(x) for x in str(score).split("-"))
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

    records = []
    skipped = 0
    params = None
    base_rates = None
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
                "p_home": pred["home_win"],
                "p_draw": pred["draw"],
                "p_away": pred["away_win"],
                "b_home": base_rates["home_win"],
                "b_draw": base_rates["draw"],
                "b_away": base_rates["away_win"],
            }
        )

    return pd.DataFrame(records), skipped


def _probs_and_onehot(results, cols):
    probs = results[cols].to_numpy(dtype=float)
    y = results["actual"].map(_OUTCOME_IDX).to_numpy()
    onehot = np.eye(3)[y]
    return probs, onehot, y


def score(results, cols):
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


def calibration_table(results, cols, bins=10):
    """Pool all predicted probabilities (across the 3 outcomes), bin them, and
    compare mean predicted probability to observed frequency in each bin."""
    probs, onehot, _ = _probs_and_onehot(results, cols)
    p_flat = probs.reshape(-1)
    e_flat = onehot.reshape(-1)

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

    model = score(results, ["p_home", "p_draw", "p_away"])
    baseline = score(results, ["b_home", "b_draw", "b_away"])

    print(f"Tested {len(results)} matches ({skipped} skipped: team not in the fit at prediction time)\n")

    metrics = pd.DataFrame(
        {
            "metric": ["log_loss", "rps", "brier", "accuracy"],
            "model": [model["log_loss"], model["rps"], model["brier"], model["accuracy"]],
            "base_rate": [baseline["log_loss"], baseline["rps"], baseline["brier"], baseline["accuracy"]],
        }
    )
    metrics["delta"] = metrics["model"] - metrics["base_rate"]
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  (lower is better for log_loss / rps / brier; higher for accuracy)\n")

    print(f"Calibration ({args.bins} bins, all outcomes pooled):")
    cal = calibration_table(results, ["p_home", "p_draw", "p_away"], bins=args.bins)
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.csv_out:
        results.to_csv(args.csv_out, index=False)
        print(f"\nWrote per-match predictions to {args.csv_out}")


if __name__ == "__main__":
    main()
