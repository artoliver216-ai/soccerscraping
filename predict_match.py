"""Compute Dixon-Coles match outcome probabilities for two teams, loading
fitted parameters from model_params.json.
"""

import argparse
import json

import numpy as np
from scipy.stats import poisson

PARAMS_PATH = "model_params.json"
MAX_GOALS = 9  # 0..9 -> a 10x10 scoreline grid


def tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(lam, mu, rho, max_goals=MAX_GOALS):
    goals = np.arange(0, max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

    for x in (0, 1):
        for y in (0, 1):
            matrix[x, y] *= tau(x, y, lam, mu, rho)

    return matrix / matrix.sum()  # renormalize after the tau adjustment


def outcome_probs(matrix):
    return {
        "home_win": np.tril(matrix, -1).sum(),
        "draw": np.trace(matrix),
        "away_win": np.triu(matrix, 1).sum(),
    }


def over_under_probs(matrix, line=2.5):
    n = matrix.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    over = matrix[totals > line].sum()
    return {"over": over, "under": 1 - over}


def asian_handicap_probs(matrix, lines=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)):
    n = matrix.shape[0]
    home_goals, away_goals = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    margin = home_goals - away_goals

    results = {}
    for line in lines:
        adj = margin + line
        results[line] = {
            "home_covers": matrix[adj > 0].sum(),
            "push": matrix[adj == 0].sum(),
            "away_covers": matrix[adj < 0].sum(),
        }
    return results


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return json.load(f)


def predict(home_team, away_team, params):
    """Return expected goals and full probability breakdown for a fixture."""
    teams = params["teams"]
    missing = [t for t in (home_team, away_team) if t not in teams]
    if missing:
        raise KeyError(f"Unknown team(s): {', '.join(missing)}. Known teams: {sorted(teams)}")

    home, away = teams[home_team], teams[away_team]
    gamma, rho = params["home_advantage"], params["rho"]

    lam = np.exp(home["attack"] + away["defense"] + gamma)
    mu = np.exp(away["attack"] + home["defense"])

    matrix = score_matrix(lam, mu, rho)
    return {
        "lambda": lam,
        "mu": mu,
        "matrix": matrix,
        **outcome_probs(matrix),
        **{f"goals_{k}": v for k, v in over_under_probs(matrix).items()},
        "handicaps": asian_handicap_probs(matrix),
    }


def main():
    parser = argparse.ArgumentParser(description="Dixon-Coles match probability predictor")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--params", default=PARAMS_PATH)
    args = parser.parse_args()

    params = load_params(args.params)
    try:
        result = predict(args.home, args.away, params)
    except KeyError as e:
        raise SystemExit(str(e))

    lam, mu = result["lambda"], result["mu"]
    outcome = {k: result[k] for k in ("home_win", "draw", "away_win")}
    goals = {"over": result["goals_over"], "under": result["goals_under"]}
    handicaps = result["handicaps"]

    print(f"{args.home} (home) vs {args.away} (away)")
    print(f"Expected goals: {args.home}={lam:.2f}, {args.away}={mu:.2f}\n")

    print("1X2:")
    print(f"  Home Win: {outcome['home_win'] * 100:.1f}%")
    print(f"  Draw:     {outcome['draw'] * 100:.1f}%")
    print(f"  Away Win: {outcome['away_win'] * 100:.1f}%\n")

    print("Total Goals:")
    print(f"  Over 2.5:  {goals['over'] * 100:.1f}%")
    print(f"  Under 2.5: {goals['under'] * 100:.1f}%\n")

    print("Asian Handicap (home line):")
    for line, probs in handicaps.items():
        push_txt = f", Push {probs['push'] * 100:.1f}%" if probs["push"] > 0 else ""
        print(
            f"  {line:+.2f}: Home covers {probs['home_covers'] * 100:.1f}%, "
            f"Away covers {probs['away_covers'] * 100:.1f}%{push_txt}"
        )


if __name__ == "__main__":
    main()
