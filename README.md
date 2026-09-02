# Soccer Scraping & Dixon-Coles Predictor

A small pipeline that scrapes Premier League xG data, fits a time-weighted
Dixon-Coles model to it, and predicts match outcome probabilities.

```
scrape_xg.py  --->  fbref_xg.csv  --->  fit_dixon_coles.py  --->  model_params.json  --->  predict_match.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests beautifulsoup4 pandas scipy playwright
playwright install chromium
```

## 1. `scrape_xg.py`

Scrapes completed Premier League matches (2024/25 and 2025/26 seasons) from
[Understat](https://understat.com), including each team's expected goals
(xG) per match.

Understat's match calendar is rendered client-side with JavaScript, so this
uses Playwright to drive a real (headless) Chromium browser rather than
plain `requests`. FBref was the original target, but it sits behind a
Cloudflare bot challenge that blocks scripted requests, so this scrapes
Understat instead.

For each season, the script starts at the most recent completed week and
clicks "prev week" repeatedly, collecting matches until the button
disables (start of season), inserting a 3-second delay before each click
to stay under rate limits.

```bash
python scrape_xg.py
```

Output: `fbref_xg.csv` with columns `Season, Date, Home, Home_xG, Score, Away_xG, Away`.

## 2. `fit_dixon_coles.py`

Fits a [Dixon-Coles](https://www.jstor.org/stable/2986284) model — attack
rating (α) and defense rating (β) per team, a home-advantage term (γ), and
a low-score correlation adjustment (ρ) — using **Home_xG/Away_xG in place
of actual goals**.

- **Time-decay weighting**: each match is weighted `exp(-ξ * days_ago)`
  with `ξ = 0.00231` (the value from the original Dixon & Coles paper),
  anchored to the most recent match date, so recent form counts more.
- **Identifiability constraint**: the Dixon-Coles likelihood is invariant
  to shifting every team's attack by `+c` and defense by `-c`. Rather than
  constraining the optimizer directly, the model fits unconstrained
  (`scipy.optimize.minimize`, L-BFGS-B) and then shifts the result so that
  `sum(attack) = 0` — equivalent to a hard constraint, but numerically
  more stable.
- **ρ is bounded to `[-0.15, 0.15]`**, the historically typical range.
  Because xG is continuous, the low-score correction (which the original
  model applies only to exact scorelines 0-0/1-0/0-1/1-1) is evaluated
  here on *rounded* xG values. This is an approximation — the correction
  doesn't correspond to a real statistical anomaly the way it does for
  actual goal counts, so the fit tends to push ρ toward whatever bound is
  set rather than settling on it naturally. Treat ρ as a soft regularizer,
  not a strongly-identified parameter.

```bash
python fit_dixon_coles.py
```

Prints a summary table of fitted α/β per team plus γ and ρ, and saves all
parameters to `model_params.json`.

## 3. `predict_match.py`

Loads `model_params.json` and computes match probabilities for a given
fixture.

```bash
python predict_match.py --home Arsenal --away Chelsea
```

Steps:
1. Computes expected goals `λ = exp(α_home + β_away + γ)` and
   `μ = exp(α_away + β_home)`.
2. Builds a 10×10 Poisson scoreline grid (0–9 goals each side), applying
   the Dixon-Coles τ(ρ) correction to the four low-score cells, then
   renormalizes.
3. Sums grid cells into:
   - **1X2**: Home Win / Draw / Away Win probabilities
   - **Over/Under 2.5 goals**
   - **Asian handicap** probabilities across lines -1.5 to +1.5

## Data files

- `fbref_xg.csv` — scraped match data (output of step 1).
- `model_params.json` — fitted model parameters (output of step 2, input
  to step 3): per-team `attack`/`defense`, plus `home_advantage`, `rho`,
  and `xi`.

Both are committed so `predict_match.py` can be run without re-scraping or
re-fitting.
