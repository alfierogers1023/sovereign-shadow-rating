# Sovereign Shadow-Rating Model

A fundamentals-driven sovereign credit "shadow rating" engine. It scores ~40
sovereigns from macro / fiscal / external / governance indicators, predicts a
credit rating, and compares the prediction to the actual S&P / Moody's / Fitch
ratings. **The analytical payload is the divergence** — where the model and the
agencies disagree, and why.

This repo is built to run from the **terminal**, edited in **VS Code**. No
notebook required.

---

## Quick start

```bash
# 1. From the project root, create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install Phase 0 dependencies
pip install -r requirements.txt

# 3. (Optional) enable the FRED market-spread cross-check
cp .env.example .env               # then paste your free FRED key into .env
export FRED_API_KEY=your_key_here  # or load .env however you prefer

# 4. Run Phase 0
python -m shadowrating.cli phase0
```

In VS Code: open the folder, select the `.venv` interpreter, and either run the
command above in the integrated terminal or hit **F5** (a "Phase 0" debug config
is included).

---

## Commands

| Command | What it does |
|---|---|
| `python -m shadowrating.cli ratings` | Show the target ratings table (letters → notches → consensus, IG/HY flag, split ratings) |
| `python -m shadowrating.cli wdi` | Pull World Bank WDI indicators |
| `python -m shadowrating.cli wgi` | Pull World Bank WGI governance indicators |
| `python -m shadowrating.cli weo` | Parse the IMF WEO flat file (see below) |
| `python -m shadowrating.cli fred` | Pull FRED long-term bond yields + spreads |
| `python -m shadowrating.cli phase0` | Run all loaders, assemble the panel, print the coverage report |
| add `--fresh` to any command | Ignore caches and re-pull from source |

Each loader caches to `data/raw/*.parquet`, so reruns are instant and offline.

---

## The one manual step: IMF WEO

There's no clean free IMF API, so the WEO is loaded from a flat file you
download once:

1. Open <https://www.imf.org/en/Publications/WEO/weo-database> (latest release).
2. Download the **"By Countries" → full "all" tab-delimited dataset**.
3. Save it as `data/raw/weo.tsv`.

If the file isn't there, Phase 0 still runs — it just reports WEO as missing and
shows you these instructions. World Bank (WDI + WGI) and FRED are fully API-driven
and need no manual download.

---

## What Phase 0 produces

- `data/ratings.csv` — the **target**: current S&P / Fitch / Moody's long-term
  foreign-currency ratings for ~42 sovereigns, snapshot-dated. Verify or re-pull
  before relying on it; ratings drift.
- `data/raw/wdi.parquet`, `wgi.parquet`, `weo.parquet`, `fred_yields.parquet` —
  cached source pulls.
- `data/raw/panel.parquet` — the assembled country × indicator panel.
- A printed **coverage report** showing exactly which country/indicator cells
  are missing. Coverage gaps cluster in the distressed names — they are reported,
  never silently dropped.

---

## Layout

```
sovereign-shadow-rating/
├── README.md
├── requirements.txt
├── data/
│   ├── ratings.csv            # target variable (snapshot-dated)
│   └── raw/                   # cached pulls + the WEO download go here
└── shadowrating/
    ├── config.py             # country sample, indicator codes, weights, paths
    ├── ratings.py            # letter↔notch mapping, consensus, IG boundary
    ├── coverage.py           # panel assembly + honest coverage reporting
    ├── cli.py                # entry point (python -m shadowrating.cli ...)
    └── loaders/
        ├── worldbank.py      # WDI + WGI via wbgapi
        ├── imf_weo.py        # WEO flat-file parser
        └── fred.py           # bond yields + spreads
```

## Design principles (kept through every phase)

1. **Validate against the real world.** Every output is benchmarked against
   actual agency ratings (and market spreads where available). Report
   cross-validated error, never in-sample fit.
2. **Two models, not one.** A transparent rules-based scorecard *and* a
   statistical model. The comparison — and the gap vs actuals — is the product.
3. **Productise and publish.** The end state is a live, dated, hosted dashboard,
   not a static PDF.

## Roadmap

- [x] **Phase 0** — scaffolding, loaders, ratings target, coverage
- [ ] Phase 1 — scaling (percentile-rank, signed) + pillar features
- [ ] Phase 2 — Model A: rules-based scorecard
- [ ] Phase 3 — Model B: ordered logit + gradient boosting, LOOCV
- [ ] Phase 4 — validation + divergence table + market cross-check
- [ ] Phase 5 — DSA scenario module + fan charts
- [ ] Phase 6 — Chart.js dashboard, host on GitHub Pages

## Caveats baked in from the spec

- Cross-sectional snapshot (one observation per country), not a panel — stated
  plainly, not hidden.
- Small N (~42): ordered logit is the honest headline model; tree importances
  are descriptive, not predictive.
- Free data is revised, not point-in-time — there's vintage/look-ahead slack.
- Scaling is sample-relative, so the scorecard→notch calibration won't transfer
  cleanly to out-of-sample observations without recalibration.
