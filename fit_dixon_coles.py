"""Fit a time-weighted Dixon-Coles model to fbref_xg.csv, using Home_xG/Away_xG
in place of actual goals as the model's target values.

Reference: Dixon, M.J. and Coles, S.G. (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market."

Because xG is continuous rather than integer goal counts, two adaptations are
made to the standard Dixon-Coles formulation:
  - The Poisson log-likelihood is generalized to non-integer counts via the
    log-gamma function (k! -> Gamma(k+1)).
  - The low-score correction (tau) is evaluated on the *rounded* xG values,
    since it is defined only for the scorelines 0-0, 1-0, 0-1, 1-1.
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

CSV_PATH = "fbref_xg.csv"
PARAMS_PATH = "model_params.json"
XI = 0.00231  # time-decay rate (Dixon & Coles 1997)


def load_matches(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%A, %B %d, %Y")
    return df.sort_values("Date").reset_index(drop=True)


def build_weights(dates, xi):
    """Exponential time-decay weights, anchored to the most recent match date."""
    days_ago = (dates.max() - dates).dt.days.to_numpy()
    return np.exp(-xi * days_ago)


def tau(x, y, lam, mu, rho):
    xi, yi = int(round(x)), int(round(y))
    if xi == 0 and yi == 0:
        return 1 - lam * mu * rho
    if xi == 0 and yi == 1:
        return 1 + lam * rho
    if xi == 1 and yi == 0:
        return 1 + mu * rho
    if xi == 1 and yi == 1:
        return 1 - rho
    return 1.0


def poisson_logpmf(k, rate):
    return -rate + k * np.log(rate) - gammaln(k + 1)


def negative_log_likelihood(params, n_teams, home_idx, away_idx, home_xg, away_xg, weights):
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    gamma, rho = params[-2], params[-1]

    lam = np.exp(attack[home_idx] + defense[away_idx] + gamma)
    mu = np.exp(attack[away_idx] + defense[home_idx])

    ll = poisson_logpmf(home_xg, lam) + poisson_logpmf(away_xg, mu)
    tau_vals = np.array(
        [np.log(max(tau(hx, ax, l, m, rho), 1e-10)) for hx, ax, l, m in zip(home_xg, away_xg, lam, mu)]
    )
    return -np.sum(weights * (ll + tau_vals))


def fit(df, xi=XI):
    teams = sorted(set(df["Home"]) | set(df["Away"]))
    team_index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = df["Home"].map(team_index).to_numpy()
    away_idx = df["Away"].map(team_index).to_numpy()
    home_xg = df["Home_xG"].to_numpy(dtype=float)
    away_xg = df["Away_xG"].to_numpy(dtype=float)
    weights = build_weights(df["Date"], xi)

    x0 = np.concatenate([np.zeros(n_teams), np.zeros(n_teams), [0.2], [-0.1]])
    bounds = [(-3, 3)] * n_teams + [(-3, 3)] * n_teams + [(-2, 2), (-0.15, 0.15)]

    result = minimize(
        negative_log_likelihood,
        x0,
        args=(n_teams, home_idx, away_idx, home_xg, away_xg, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    attack = result.x[:n_teams]
    defense = result.x[n_teams : 2 * n_teams]
    gamma, rho = result.x[-2], result.x[-1]

    # The log-likelihood is invariant to shifting all attacks by +c and all
    # defenses by -c, so we can normalize post-hoc to enforce sum(attack) = 0
    # for identifiability without constraining the optimizer directly.
    shift = attack.mean()
    attack = attack - shift
    defense = defense + shift

    return teams, attack, defense, gamma, rho, result


def main():
    df = load_matches(CSV_PATH)
    teams, attack, defense, gamma, rho, result = fit(df)

    summary = pd.DataFrame(
        {"Team": teams, "Attack (alpha)": attack, "Defense (beta)": defense}
    ).sort_values("Attack (alpha)", ascending=False)

    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nHome Advantage (gamma): {gamma:+.4f}")
    print(f"Low-Score Adjustment (rho): {rho:+.4f}")
    print(f"Sum of attack ratings: {attack.sum():.2e}")
    print(f"Log-likelihood: {-result.fun:.2f}")

    params = {
        "teams": {
            team: {"attack": float(a), "defense": float(b)}
            for team, a, b in zip(teams, attack, defense)
        },
        "home_advantage": float(gamma),
        "rho": float(rho),
        "xi": XI,
    }
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved parameters to {PARAMS_PATH}")


if __name__ == "__main__":
    main()
