# CLAUDE.md — Sovereign Shadow-Rating Model

> This file is read automatically by Claude Code at the start of every session.
> It is the single source of truth for what this project is, where it stands, and
> what to do next. If you are a fresh Claude Code instance: read this top to
> bottom before touching anything, then see **"How to start a session"** below.

---

## What this project is

A fundamentals-driven sovereign credit **shadow-rating** engine. It scores ~42
sovereigns from macro / fiscal / external / governance indicators, predicts a
credit rating, and compares the prediction to the actual S&P / Moody's / Fitch
ratings. **The analytical payload is the divergence** — where the model and the
agencies disagree, and why. Two later modules extend it: an early-warning layer
and an IMF-style debt-sustainability (DSA) scenario tool. The end state is a
live, hosted Chart.js dashboard on GitHub Pages.

It runs from the **terminal**, edited in **VS Code**. No notebooks. The owner
prefers modular, terminal-runnable Python packages.

## Three design principles — preserve these through every phase

1. **Validate against the real world.** Every output is benchmarked against
   actual agency ratings (and market spreads where available). Report
   cross-validated error (leave-one-country-out), never in-sample fit.
2. **Two models, not one.** A transparent rules-based scorecard *and* a
   statistical model. The comparison — and the gap vs actuals — is the product,
   not a single accuracy score.
3. **Productise and publish.** Public, dated, honest artifact (hosted dashboard),
   not a static PDF.

The governance / political pillar is the differentiating edge — go deeper there
than the agencies do; don't treat it as one bolt-on score.

---

## Current state — Phase 6 is COMPLETE (dashboard built; not yet published)

The `shadowrating` package is scaffolded and working:

- `data/ratings.csv` — the **target variable**: current S&P/Fitch/Moody's
  long-term FC ratings for 42 sovereigns, snapshot-dated 2026-06-25 (pulled from
  Wikipedia's list). 27 IG, 15 HY, 31 split-rated.
- `shadowrating/config.py` — the 42-country sample, every WDI/WGI/WEO/FRED
  indicator code with `pillar` + `direction` flags, pillar weights, paths.
- `shadowrating/ratings.py` — letter→notch map (AAA/Aaa=21 … D/RD/C=1),
  three-agency consensus, IG-boundary flag (notch 12 = BBB-/Baa3), split detection.
- `shadowrating/loaders/` — `worldbank.py` (WDI+WGI via wbgapi), `imf_weo.py`
  (flat file if present in `data/raw/`, else falls back automatically to IMF's
  DataMapper API — see note below), `fred.py` (bond yields + spreads). Each
  returns a tidy `iso3 | indicator | year | value` frame and caches to
  `data/raw/*.parquet`.
- `shadowrating/coverage.py` — panel assembly + honest coverage reporting.
- `shadowrating/features.py` — Phase 1: signed percentile-rank scaling per
  indicator, then pillar-level aggregation (mean of available indicators per
  pillar per country) and a weight-renormalised composite score. Pillars with
  zero indicators in the current panel (currently `fiscal`, until WEO is
  loaded) are excluded rather than zero-filled; per-country missing indicators
  within a populated pillar are excluded from that country's pillar mean and
  counted, not imputed with a guessed value.
- `shadowrating/scorecard.py` — Phase 2 / Model A: puts the unitless [0, 1]
  composite onto the agency notch scale (1..21) via a single least-squares
  line (`notch ~ a + b*composite`) — a unit conversion, not a fitted weighting
  model; pillar weights stay config-driven and fixed. Headline metric is
  leave-one-country-out (LOOCV) error, never in-sample fit, per principle 1.
  Also computes the per-country divergence table and flags IG/HY boundary
  mismatches.
- `shadowrating/model_b.py` — Phase 3 / Model B: ordered logit (`mord.LogisticAT`)
  on a 6-band rating scale (`ratings.RATING_BANDS` — fitting 20 cut points
  from N~42 isn't honest, 5 is defensible) plus shallow gradient boosting
  (`xgboost.XGBRegressor`, max_depth=2) on the continuous notch scale. Both
  use pillar scores (not raw indicators) as features and are evaluated only
  via LOOCV. Unlike Model A, these let the data choose pillar weights —
  that's the actual methodological contrast the "two models" principle is
  about. Feature importances are reported but explicitly labelled descriptive,
  not predictive, per CLAUDE.md. The ordered logit also returns a full 6-band
  predicted probability distribution and a 90% prediction interval per
  country (see the uncertainty decision below) — point estimates alone
  understate how unsure this model usually is.
- `shadowrating/validation.py` — Phase 4: combines Model A and Model B's LOOCV
  predictions into one master divergence table per country, flags countries
  where *both* models diverge from the agencies by >2 notches in the *same*
  direction (the strongest version of the signal — two differently-built
  models, one consistent disagreement), and cross-checks divergence direction
  against FRED bond-spread residuals (spread minus what the agency notch alone
  would predict) where FRED data is available. Skips the market cross-check
  cleanly (not silently) when `FRED_API_KEY` isn't set.
- `shadowrating/data_quality.py` — signal-vs-artefact rigor pass: per-country
  data vintage (`excess_age` — an indicator's age vs. the sample median for
  that same indicator, which isolates a country-specific gap from a source's
  universal publication lag) and missing-share (share of pillar/indicator
  slots absent, never filled with a guessed value). Joined onto the Phase 4
  divergence table and exported to the dashboard so a reader can tell a real
  divergence from one resting on stale or thin data.
- `shadowrating/dsa.py` — Phase 5: IMF-style debt sustainability analysis.
  Standard debt dynamics identity `d_t = d_{t-1}*(1+i)/(1+g) - pb` on the
  *raw* WEO panel (actual % of GDP levels, not the percentile-rank features).
  The effective interest rate isn't published directly — it's backed out from
  `interest_bill = primary_balance - overall_balance`, then
  `i = interest_bill / (debt_gdp/100)`. Produces: a baseline path (10y,
  current pb/i/g held constant), 3 single-factor stress tests + 1 combined
  adverse scenario (shock sizes are illustrative IMF-DSA-style magnitudes, NOT
  estimated from each country's own historical volatility — we only have one
  WEO vintage per indicator, not a time series), and a Monte Carlo fan chart
  (2000 sims/country, iid annual shocks to growth/rate/primary balance,
  10/25/50/75/90th percentile bands). All three caveats are stated in the
  module docstring and CLI output, not buried.
- `shadowrating/export_dashboard.py` — Phase 6: re-runs the full pipeline (cheap)
  and writes flat JSON to `docs/data/` (`countries.json`, `summary.json`,
  `dsa.json`, `dsa_fragility.json`) so the static dashboard always reflects one
  consistent run, never a mix of stale cached parquet files.
- `docs/` — the static Chart.js dashboard itself: `index.html` + `style.css` +
  `app.js`, no build step, Chart.js loaded from the jsdelivr CDN. Sortable
  per-country divergence table; click a row (or load `?select=ISO3`) for a
  detail panel with a pillar-score bar chart, an actual-vs-Model-A-vs-Model-B
  notch comparison, and the DSA debt/GDP chart (baseline + combined-adverse
  scenario lines + Monte Carlo p10–p90 fan). Verified by rendering it in
  headless Chrome and screenshotting — table, sorting, and all three detail
  charts confirmed working visually, not just "the JSON parses."
- `shadowrating/cli.py` — `python -m shadowrating.cli {ratings,wdi,wgi,weo,fred,phase0,phase1,phase2,phase3,phase4,phase5,phase6}`,
  plus a `--fresh` flag to bypass caches.

The `ratings`, `phase0`, `phase1`, `phase2`, `phase3`, and `phase4` commands
all run end-to-end with **all four data sources live**: WDI, WGI, WEO (via the
DataMapper API fallback — no local flat file on this machine), and FRED. Full
panel: 42 countries × 18 indicators, 100% fill rate, all five pillars
populated (`economic`, `external`, `fiscal`, `institutional`, `monetary`).

**Below numbers are from the run with fiscal data included** — they
supersede earlier estimates from when `fiscal` was an empty pillar.

**Phase 2 result (post leakage-fix, current):** LOOCV MAE = 3.00 notches,
RMSE = 3.93, bias = -0.27. 10 IG/HY boundary mismatches. Biggest divergences:
Sri Lanka (+9.6), Ghana (+9.1), USA (model BBB-, actual AA+, **-8.5**),
Belgium (-7.0). Adding the fiscal pillar did *not* shrink Sri Lanka's gap as
hypothesized — it got slightly worse — and surfaced a new, larger one: Model A
penalizes the US hard for fiscal metrics (high debt/GDP, persistent deficit)
that the agencies don't punish nearly as much, almost certainly because pure
percentile-rank scoring can't see reserve-currency status / unique funding
advantages. That's a real, structural blind spot worth stating plainly in the
write-up, not something to engineer away. (Pre-fix numbers were MAE 2.96,
RMSE 3.92, bias -0.07 — the fold-wise-scaling fix above moved these slightly,
as expected from a real-but-small leak, not a different conclusion.)

**Phase 3 result (post leakage-fix, current):** ordered logit LOOCV — 52%
exact band match, 88% within one band, MAE 0.67 bands. Gradient boosting
LOOCV — MAE 1.88 notches, RMSE 2.64, bias -0.25 (still beats Model A). GBM
feature importance with fiscal included (descriptive only): institutional
0.58 ≫ monetary 0.24 > economic 0.09 ≈ external 0.05 ≈ **fiscal 0.03** — the
tree model effectively decided fiscal score, in isolation, isn't very
predictive of the agency notch (consistent with the US finding above: fiscal
alone is a weak/confounded signal). Sri Lanka is still badly misclassified by
the ordered logit (predicted "A" band vs actual CCC-D). (Pre-fix: 57%/88%/0.62
bands, GBM MAE 1.95/RMSE 2.60/bias -0.02.) **Uncertainty check (new):** the
ordered logit's 90% prediction interval actually contains the truth 98% of
the time across the sample — it's underconfident, not overconfident; most of
its apparent "misses" are the model honestly being unsure, not being wrong
with conviction. Sri Lanka is the single exception: the model's own 90%
interval excludes the actual band, the only country in the sample where that
happens.

**Phase 4 result (post leakage-fix, current):** 11 countries where Model A and
Model B's GBM diverge from the agencies by >2 notches in the *same* direction:
Sri Lanka, Ghana, USA, Belgium, China, Argentina, Colombia, Romania, Mexico,
Australia, Germany. The market-spread cross-check is live (FRED pulled
successfully) — for the 21 FRED-covered (mostly OECD) countries, Model A's
divergence direction is corroborated by bond-spread residuals in 13/21 cases,
including the three largest divergences in that subset (Mexico, Australia,
USA all have wider spreads than their agency notch alone would predict — some
independent market support for those countries being overrated by the
agencies relative to fundamentals, exactly where Model A also flags them).
Of the 11 same-direction-flagged countries, only **Sri Lanka** also carries a
data-vintage red flag (+2 years excess age, spanning 4 of its 5 pillars) —
every other flagged divergence (Ghana, USA, Belgium, China, Argentina,
Colombia, Romania, Mexico, Australia, Germany) rests on data no staler than
the sample norm, so staleness can be ruled out as the explanation for those.
**Adding the uncertainty check converges on the same single name a third way:**
of the 11 same-direction-flagged countries, only Sri Lanka also has its
actual rating fall outside the ordered logit's own 90% interval
(`confident_divergence` = true for exactly 1 country). Three independent
methods — point-estimate agreement, data-vintage staleness, and predictive-
interval miss — single out the same country; nothing else in the sample gets
flagged by more than one of the three. That convergence, not the raw notch
MAE, is the strongest evidence in this whole project and the one fact worth
leading with in any write-up.

**Fixed/built this session:**
- World Bank renamed WGI indicator codes from bare codes (`GE.EST`, …) to
  `GOV_WGI_`-prefixed codes (`GOV_WGI_GE.EST`, …) — updated in `config.py`.
- IMF's WEO database page is JS-rendered, so there's no scriptable direct
  download link; `imf_weo.py` now falls back to IMF's official DataMapper API
  (`https://www.imf.org/external/datamapper/api/v1/<code>`) when no local flat
  file is found in `data/raw/`. One indicator's code differs between the two
  channels (primary balance: flat-file `GGXONLB_NGDP` vs. DataMapper
  `GGXONLB_G01_GDP_PT`) — mapped via `config.WEO_DATAMAPPER_CODE_OVERRIDES`.
  DataMapper returns forecast years out to ~2030; the loader caps at the
  current calendar year so this stays a same-period snapshot, not a mix of
  actuals and projections.
- `fredapi` calls `urllib` directly (unlike the other loaders, which go
  through `requests`) and the python.org macOS build doesn't wire `urllib` up
  to a CA bundle by default — this surfaced as a `CERTIFICATE_VERIFY_FAILED`
  error that looks like a network problem but isn't. Fixed by pointing
  `SSL_CERT_FILE` at `certifi`'s bundle in `shadowrating/__init__.py`, so it's
  set before any loader runs, on any machine.
- `worldbank.py`'s WDI/WGI loader was silently returning `year=None` for
  every single observation. Root cause: `wb.data.DataFrame(..., mrnev=1)`
  drops the vintage year from its output entirely when `mrnev` is used,
  while the lower-level `wb.data.fetch()` generator returns it correctly in
  every mode. Switched `_fetch` to use `wb.data.fetch()` directly. This had
  been silently breaking the data-vintage feature (see the new "signal vs.
  artefact" decision above) since whatever session first wrote that loader —
  worth a reminder that "the API call succeeds and returns plausible-looking
  values" doesn't mean every column it returns is populated correctly.
- `FRED_API_KEY` is set in `.env` (gitignored, not committed). Run
  `set -a && source .env && set +a` before `python -m shadowrating.cli ...` in
  a fresh shell, or `export FRED_API_KEY=...` directly.

**Phase 5 result:** 18/42 countries have debt/GDP still rising under the
baseline (no-shock) scenario after 10 years — including USA, France, Belgium,
Brazil, South Africa, Ukraine. Most fragile by combined-adverse-vs-baseline
gap at year 10: Japan (170.7% baseline → 247.5% adverse), Italy (123.9% →
187.2%), France (134.9% → 194.8%), Ukraine (196.3% → 256.1%). Mexico has by
far the largest r-g gap (effective rate 9.6% vs nominal growth 5.6%, a 4pp
gap) — its debt/GDP grinds upward even on an unchanged primary balance, a
genuinely different risk profile than its current BBB rating implies on a
pure debt-dynamics view (this is a leverage-trajectory flag, not the same
thing as Phase 4's rating divergence). Japan's baseline path *declines*
despite the highest debt/GDP in the sample (204%) because its implied
effective rate is near zero (0.1%) — a real artifact of extrapolating today's
ultra-low financing conditions forward 10 years unchanged; worth flagging in
the write-up as a baseline-assumption limitation, not a finding to take at
face value.

**Phase 6 result:** dashboard built (`docs/`) and visually verified in headless
Chrome — summary cards, sortable divergence table, and all three detail charts
(pillars, notch comparison, DSA fan) render correctly. **Not yet hosted**:
there's no git repository on this machine at all, so there's nothing to push
to GitHub Pages from. Publishing means `git init`, creating a GitHub repo,
pushing, and enabling Pages (Settings → Pages → serve from `/docs` on `main`)
— all visible/shared-state actions that need your explicit go-ahead, not
something to do automatically. See "Publishing to GitHub Pages" below.

---

## Roadmap

- [x] **Phase 0** — scaffolding, loaders, ratings target, coverage report
- [x] **Phase 1** — assemble the indicator panel, scale it, build pillar features
- [x] **Phase 2** — Model A: rules-based scorecard + per-country pillar table
- [x] **Phase 3** — Model B: ordered logit (statsmodels/mord) + gradient boosting, LOOCV
- [x] **Phase 4** — validation + divergence table + market-spread cross-check
- [x] **Phase 5** — DSA scenario module + debt-path fan charts
- [x] **Phase 6** — Chart.js dashboard built and verified locally; **not yet
      published** — this machine has no git repo / GitHub remote yet. See
      "Publishing to GitHub Pages" below.

## Key decisions to preserve (don't relitigate without good reason)

- **Scaling = percentile rank, signed.** Rank-based is more robust to this data's
  fat tails than raw z-scores. Apply each indicator's `direction` flag from
  config so higher always = more creditworthy (invert debt, inflation, etc.).
  Implemented in `features.scale_panel`.
- **Scaling is sample-relative** — calibration won't transfer to out-of-sample
  observations without recalibration. Note this; don't pretend otherwise.
- **Ordered logit is the honest headline model.** N≈42 is small; treat XGBoost
  feature importances as *descriptive*, not predictive. Keep trees shallow.
  Implemented on a 6-band scale (`ratings.RATING_BANDS`), not the full
  21-notch scale — fitting 20 ordered-logit thresholds from 42 points would
  not be honest, fitting 5 is.
- **LOOCV, not in-sample.** Report the out-of-sample notch-error distribution.
  Done for both Model A (`scorecard.loocv_predict`) and Model B
  (`model_b.fit_ordered_logit_loocv` / `fit_gbm_loocv`).
- **LOOCV must also be fold-wise in *feature engineering*, not just model
  fitting.** Found during a rigor pass: percentile-rank scaling computed once
  on the full 42-country panel (as `features.build_features` does) leaks mild,
  X-only (not target) information into a number reported as out-of-sample — a
  held-out country's rank reflects a distribution that includes itself.
  Measured the size directly before deciding whether to fix it: mean 3.2
  percentage points, max 4.7pp shift per indicator across all 42 countries.
  Fixed properly rather than just disclosed: `features.loocv_folds` rebuilds
  percentile-rank scaling from scratch on the 41 training countries every
  fold, and scales the held-out country's raw indicators against that
  training distribution only. `scorecard.loocv_predict` and
  `model_b.fit_ordered_logit_loocv`/`fit_gbm_loocv` all consume this now
  instead of a precomputed full-sample `pillar_scores`/`composite`. Effect on
  headline numbers was small and in the expected direction, confirming the
  leak was real but minor: Model A MAE 2.96→3.00, Model B GBM MAE 1.95→1.88,
  ordered logit exact-band-match 57%→52%. `features.build_features` (full-
  sample scaling) is still correct to use for *descriptive* display only —
  e.g. the dashboard's pillar bar chart for one country — never for a number
  reported as validated/out-of-sample.
- **Signal vs. artefact: surface data vintage and missing-share alongside
  every divergence, don't just report the divergence.** A real finding from
  this rigor pass: WDI/WGI's loader (`worldbank.py`) was silently returning
  `year=None` for every cell because `wb.data.DataFrame(..., mrnev=1)` drops
  the vintage year — switched to `wb.data.fetch()` (the lower-level
  generator), which returns it correctly in both mrnev and pinned-vintage
  modes. Once vintage was actually available, the useful metric turned out
  *not* to be absolute age (WDI/WGI lag ~2 years for literally every country,
  so "max age" is ~2 for almost everyone and is a near-constant, not a
  signal) but **excess age**: an indicator's age minus that same indicator's
  age for the sample median (`data_quality.excess_age`). That isolates a
  country-specific data gap from a source's universal publication lag.
  Result: Sri Lanka is the only country in the sample whose entire WEO
  release (fiscal, external, economic, *and* monetary pillars) is stuck on a
  vintage two years older than every single peer; Zambia has one indicator
  one year behind. Both are exactly the names already flagged as large
  divergences in Phase 4 — this lets a reader judge whether that's a real
  disagreement with the agencies or partly an artefact of stale inputs,
  rather than taking the divergence number at face value. Also tracks
  "missing share" (share of pillar/indicator slots absent and excluded from
  that pillar's average) as the second signal-vs-artefact axis — currently
  0% everywhere since panel coverage is 100%, which is itself worth stating
  plainly rather than omitting an always-zero metric. Surfaced in
  `data_quality.py`, the Phase 4 CLI output, and the dashboard (a "Stale?"
  column in the divergence table + a data-quality note in the country detail
  panel).
- **Report uncertainty, not just point estimates — and let it gate the
  divergence flag.** `model_b.fit_ordered_logit_loocv` now returns each
  fold's full 6-band predicted probability distribution (`mord`'s
  `predict_proba`, reindexed onto the full 0..5 band range since a fold's
  `classes_` can in principle omit a band never seen in that fold's training
  set) and a 90% prediction interval (`_central_interval` — smallest
  contiguous band range whose cumulative probability covers ≥90%, rounding
  outward so the interval never claims more confidence than the model
  actually has). Point estimates are kept; this is in addition.
  `validation.build_master_table` adds `confident_divergence` — true only
  when a country is flagged by *both* the existing >2-notch point-estimate
  agreement *and* the actual rating falls outside the ordered logit's own
  90% interval. Result, and the reason this was worth building: of the 11
  countries flagged by point estimates, only **1** — Sri Lanka — also clears
  this bar. Calibration check on the full sample: the actual band falls
  inside the model's own 90% interval 98% of the time (vs. a target of 90%),
  meaning the model is *underconfident* — most of its intervals are wide
  enough to rarely be wrong, which also makes them rarely informative. Sri
  Lanka is the one country where the model is genuinely, narrowly confident
  and still misses, which is a much stronger and more specific claim than
  "11 countries diverge" — and it's the same country flagged independently
  by the data-vintage check above. Three independent angles (point-estimate
  divergence, data staleness, predictive-interval miss) now converge on the
  same single name; nothing converges this strongly on any other country.
  Surfaced in `cli.py phase3`/`phase4` output and the dashboard (a
  "Confident?" column, a calibration stat card, and a per-country predicted-
  probability bar chart with the 90% interval shaded and the actual outcome
  outlined).
- **Never silently drop a country.** Coverage gaps cluster in the distressed
  names — report them, impute transparently within pillar, document every
  imputed cell. `features.pillar_scores` returns a missing-count matrix
  alongside the scores for exactly this reason; a pillar with zero indicators
  anywhere in the panel is dropped from the composite entirely (not treated as
  100% missing) until a real data source for it is loaded.
- **Pillar weights must be justified** against a named agency methodology in the
  write-up, not invented. The weights in config are an illustrative anchor.
- **Don't expect to beat the agencies.** Ratings are slow / through-the-cycle.
  The product is the gaps, not a higher accuracy number.

---

## How to start a session

**First time on this machine — already done.** `.venv` is set up, dependencies
(including the Phase 3 modelling stack: scikit-learn, statsmodels, mord,
xgboost) are installed, `.env` has a working `FRED_API_KEY`, and `phase0`
through `phase6` all run clean with every data source live (WDI, WGI, WEO via
DataMapper, FRED). To resume on a *new* machine instead:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # paste a free FRED key from https://fredaccount.stlouisfed.org/apikeys
set -a && source .env && set +a    # or `export FRED_API_KEY=...` directly
python -m shadowrating.cli phase0
python -m shadowrating.cli phase1
python -m shadowrating.cli phase2
python -m shadowrating.cli phase3
python -m shadowrating.cli phase4
python -m shadowrating.cli phase5
python -m shadowrating.cli phase6   # exports docs/data/*.json for the dashboard
cd docs && python -m http.server 8000   # then open http://localhost:8000
```

WDI, WGI, and WEO need no manual step at all (WEO auto-falls-back to IMF's
DataMapper API). Only FRED needs a one-time free key — without it, `fred`
prints a clear skip message and Phase 4's market cross-check is omitted
cleanly rather than silently. If you want the fully-offline, pinned-vintage
WEO path instead of the live API, download the tab-delimited "all" dataset
from <https://www.imf.org/en/Publications/WEO/weo-database> and save it as
`data/raw/weo.tsv` — the loader prefers a local file over the API if one
exists.

**Every session:** the owner wants to do as little manual work as possible.
Default behavior — read this file, check the roadmap for the first unchecked
phase, and pick up there. Explain each step as you go. Run things in the
integrated terminal and show results; don't just describe them. Update the
roadmap checkboxes in this file as phases complete.

## Publishing to GitHub Pages

The dashboard is built and works locally, but **this machine has no git
repository at all yet** — `git init` was never run, so there's no commit
history and no remote to push to. None of that should happen silently: repo
creation, the first push, and enabling Pages are all visible/shared-state
actions. When the owner is ready to actually publish:

1. `git init`, commit everything except what's gitignored (`.venv/`, `.env`,
   `data/raw/*.parquet`, `data/processed/*.parquet` — `docs/data/*.json` IS
   meant to be committed, it's the published artifact, not a cache).
2. Create a GitHub repo and push.
3. In the repo's Settings → Pages, set source to the `main` branch, `/docs`
   folder.
4. Re-run `python -m shadowrating.cli phase6` and commit+push whenever the
   underlying data/models change, to keep the published dashboard in sync —
   there's no CI wiring this up automatically yet.

## Conventions

- Terminal-first, modular Python. No notebooks.
- New code goes in the `shadowrating` package; keep modules self-contained with
  clear seams. Cache expensive pulls to `data/raw/`; cache derived features to
  `data/processed/` (both gitignored — regenerable from source).
- Match the existing style in the package (type hints, short docstrings that say
  *why*, graceful degradation when a data source is unavailable).
- The owner is technical but wants the reasoning surfaced, not hidden.
