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

Finds +EV Premier League bets by comparing live bookmaker odds against
the Dixon-Coles model, across two markets: **1X2** (`h2h`) and
**Over/Under 2.5 total goals** (`totals`).

```bash
python find_ev_bets.py                          # live odds, both markets
python find_ev_bets.py --min-ev 0.05 --bankroll 500
python find_ev_bets.py --market totals           # only over/under 2.5
python find_ev_bets.py --dry-run                 # sample odds, no API key needed
```

Steps:
1. Pulls live EPL `h2h` and `totals` odds from
   [the-odds-api.com](https://the-odds-api.com) via `requests`
   (`--market` narrows the request to one of them).
2. For each fixture, calls `predict_match.predict()` to get model
   probabilities. Team names differ slightly between the odds API's full
   official names (e.g. "Tottenham Hotspur") and the model's shorter
   Understat-derived names (e.g. "Tottenham") — a `TEAM_NAME_ALIASES` map
   handles the known cases; unmatched fixtures are skipped with a warning
   rather than silently mispriced.
3. Maps each bookmaker outcome to a model probability (`resolve_outcome`):
   home/away/draw for `h2h`; Over/Under for `totals`. **Only the 2.5
   goals line is priced** — the model computes Over/Under at 2.5 by
   default — so a book's other totals lines (3.5, etc.) are ignored.
4. For each priced outcome, computes:
   - **EV** = `(model probability × decimal odds) - 1`
   - **Stake** via fractional Kelly (`f* = (p·b - q) / b`, scaled by
     `--kelly-fraction`, default 0.25 = quarter-Kelly, a common way to
     reduce variance versus full Kelly)
5. Prints every outcome with `EV ≥ --min-ev` (default `0.03` = +3%),
   with a `Market` column, sorted by EV descending.

Key flags: `--api-key` (or set `ODDS_API_KEY`), `--region`
(`uk`/`eu`/`us`/`au`), `--market` (`all`/`h2h`/`totals`), `--bankroll`,
`--min-ev`, `--kelly-fraction`, `--dry-run`.

**Caveat**: EV here is only as good as the underlying model — and since
the Dixon-Coles fit here is on xG rather than actual goals with a
loosely-identified ρ (see the caveat in `fit_dixon_coles.py` above),
treat the model probabilities as directional rather than sharp. The
backtest (below) shows the model runs over-confident in the mid-range,
and being an xG fit it tends to over-state Over 2.5 in particular.
Compare against bookmaker closing lines before trusting a signal, and
note that recommended stakes are unconstrained (no bankroll cap across
simultaneous bets) — apply your own risk limits before betting real money.

## `backtest.py`

Walk-forward calibration check for the model (not part of the pipeline —
an evaluation tool). Processes matches in date order; for each test match
it refits the model on only the matches before it, then scores its
predictions against the actual result, in two markets:

- **1X2** — log-loss, **RPS** (ranked probability score, the standard
  football forecasting metric), multiclass Brier, argmax accuracy
- **Over/Under 2.5 goals** — binary log-loss, Brier, accuracy on
  `P(Over)`

```bash
python backtest.py
python backtest.py --min-train 200 --refit-every 1 --csv-out bt.csv
```

Each metric is shown next to a base-rate baseline (the running frequency
of that outcome in the training set), and each market gets a
**calibration table** (predicted probability binned, mean predicted vs
observed frequency per bin — 1X2 pools all three outcomes).

Refitting every match is slow, so `--refit-every` (default 10) reuses a
fit for that many matches; pass `1` for a true match-by-match backtest.
On the current data the two agree to ~0.3% on every metric, so the
default 10 costs almost nothing. `--min-matches` is passed through to the
sparse-team filter. There is no ROI simulation — the odds API's free tier
has no historical closing lines to bet against.

On the committed data (`--min-train 150 --refit-every 1`, 245 matches
scored, 5 skipped) the model beats the base rate on every metric:

| metric   | model | base rate |
|----------|-------|-----------|
| log-loss | 1.039 | 1.107     |
| RPS      | 0.204 | 0.227     |
| Brier    | 0.627 | 0.672     |
| accuracy | 47.8% | 38.8%     |

RPS ≈ 0.20 is in the range published football models report. But the
calibration table shows the model is over-confident in the 0.5–0.7
probability band (predicts ~0.55 / ~0.64, observes ~0.49 / ~0.48) —
short-priced favourites and clear underdogs are well-calibrated. This is
the "directional, not sharp" caveat, quantified. `backtest_results.csv`
holds the per-match predictions from that run.

## Data files

- `fbref_xg.csv` — scraped match data (output of step 1).
- `model_params.json` — fitted model parameters (output of step 2, input
  to step 3): per-team `attack`/`defense`/`matches`, plus `home_advantage`,
  `rho`, `xi`, and `min_matches`.
- `backtest_results.csv` — per-match predicted vs base-rate probabilities
  from `backtest.py --refit-every 1` (see above).

`fbref_xg.csv` and `model_params.json` are committed so `predict_match.py`
can be run without re-scraping or re-fitting.
