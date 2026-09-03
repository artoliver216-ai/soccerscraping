# Soccer Scraping & Dixon-Coles Predictor

A small pipeline that scrapes Premier League xG data, fits a time-weighted
Dixon-Coles model to it, and predicts match outcome probabilities.

```
scrape_xg.py  --->  fbref_xg.csv  --->  fit_dixon_coles.py  --->  model_params.json  --->  predict_match.py  --->  find_ev_bets.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests beautifulsoup4 pandas scipy playwright
playwright install chromium
```

`find_ev_bets.py` also needs a free API key from
[the-odds-api.com](https://the-odds-api.com):

```bash
export ODDS_API_KEY=your_key_here
```

## 1. `scrape_xg.py`

Scrapes completed Premier League matches (2025/26 and 2026/27 seasons) from
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
- **Min-matches guard**: teams with fewer than `--min-matches` games
  (default 6) are dropped before fitting, along with every match involving
  them. A few games is too little to separate a team's attack from its
  defense, and the optimizer will fit that noise — distorting γ, ρ, and the
  ratings of everyone they played. The filter is applied iteratively
  (dropping a sparse team's matches can pull a borderline opponent under
  the threshold too). Dropped teams are simply absent from
  `model_params.json`, so `predict_match.py` / `find_ev_bets.py` skip any
  fixture involving them rather than pricing it off a bad rating. This
  mostly bites newly-promoted sides early in a season.

```bash
python fit_dixon_coles.py
python fit_dixon_coles.py --min-matches 10   # stricter
```

Prints a summary table of fitted α/β and match count per team plus γ and
ρ, and saves all parameters to `model_params.json` (per-team `attack`,
`defense`, `matches`, plus `home_advantage`, `rho`, `xi`, `min_matches`).

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

`load_params()` and `predict()` are also exposed as importable functions
(not just via the CLI), so other scripts — like `find_ev_bets.py` — can
call the model directly instead of duplicating the probability math.

## 4. `find_ev_bets.py`

Finds +EV Premier League 1X2 bets by comparing live bookmaker odds against
the Dixon-Coles model.

```bash
python find_ev_bets.py                          # live odds, default thresholds
python find_ev_bets.py --min-ev 0.05 --bankroll 500
python find_ev_bets.py --dry-run                 # sample odds, no API key needed
```

Steps:
1. Pulls live EPL 1X2 (`h2h`) odds from
   [the-odds-api.com](https://the-odds-api.com) via `requests`.
2. For each fixture, calls `predict_match.predict()` to get model
   probabilities. Team names differ slightly between the odds API's full
   official names (e.g. "Tottenham Hotspur") and the model's shorter
   Understat-derived names (e.g. "Tottenham") — a `TEAM_NAME_ALIASES` map
   handles the known cases; unmatched fixtures are skipped with a warning
   rather than silently mispriced.
3. For each bookmaker outcome, computes:
   - **EV** = `(model probability × decimal odds) - 1`
   - **Stake** via fractional Kelly (`f* = (p·b - q) / b`, scaled by
     `--kelly-fraction`, default 0.25 = quarter-Kelly, a common way to
     reduce variance versus full Kelly)
4. Prints every outcome with `EV ≥ --min-ev` (default `0.03` = +3%),
   sorted by EV descending.

Key flags: `--api-key` (or set `ODDS_API_KEY`), `--region`
(`uk`/`eu`/`us`/`au`), `--bankroll`, `--min-ev`, `--kelly-fraction`,
`--dry-run`.

**Caveat**: EV here is only as good as the underlying model — and since
the Dixon-Coles fit here is on xG rather than actual goals with a
loosely-identified ρ (see the caveat in `fit_dixon_coles.py` above),
treat the model probabilities as directional rather than sharp. Compare
against bookmaker closing lines before trusting a signal, and note that
recommended stakes are unconstrained (no bankroll cap across
simultaneous bets) — apply your own risk limits before betting real money.

## `backtest.py`

Walk-forward calibration check for the model (not part of the pipeline —
an evaluation tool). Processes matches in date order; for each test match
it refits the model on only the matches before it, then scores the
predicted Home/Draw/Away probabilities against the actual result.

```bash
python backtest.py
python backtest.py --min-train 200 --refit-every 1 --csv-out bt.csv
```

Reports, each next to a base-rate baseline (running H/D/A frequency in
the training set):

- **log-loss**, **RPS** (ranked probability score — the standard football
  forecasting metric), **multiclass Brier**, **argmax accuracy**
- a **calibration table**: predicted probabilities pooled across all three
  outcomes, binned, mean predicted vs observed frequency per bin

Refitting every match is slow, so `--refit-every` (default 10) reuses a
fit for that many matches; pass `1` for a true match-by-match backtest.
`--min-matches` is passed through to the sparse-team filter. There is no
ROI simulation — the odds API's free tier has no historical closing
lines to bet against.

On the committed data (`--min-train 150 --refit-every 10`) the model beats
the base rate on every metric (log-loss ≈ 1.04 vs 1.11, accuracy ≈ 48% vs
39%) but is visibly over-confident in the 0.5–0.7 probability band — the
same "directional, not sharp" caveat, quantified.

## Data files

- `fbref_xg.csv` — scraped match data (output of step 1).
- `model_params.json` — fitted model parameters (output of step 2, input
  to step 3): per-team `attack`/`defense`/`matches`, plus `home_advantage`,
  `rho`, `xi`, and `min_matches`.

Both are committed so `predict_match.py` can be run without re-scraping or
re-fitting.
