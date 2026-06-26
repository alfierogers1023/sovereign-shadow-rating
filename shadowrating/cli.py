"""
Command-line entry point.

Run from the project root:

    python -m shadowrating.cli ratings        # show the target ratings table
    python -m shadowrating.cli wdi            # pull World Bank WDI
    python -m shadowrating.cli wgi            # pull World Bank WGI
    python -m shadowrating.cli weo            # parse IMF WEO flat file (if present)
    python -m shadowrating.cli fred           # pull FRED yields (needs FRED_API_KEY)
    python -m shadowrating.cli phase0         # run everything + coverage matrix
    python -m shadowrating.cli phase0 --fresh # ignore caches and re-pull
    python -m shadowrating.cli phase1         # scale panel + build pillar features
    python -m shadowrating.cli phase2         # Model A scorecard + LOOCV divergence vs actuals
    python -m shadowrating.cli phase3         # Model B: ordered logit + gradient boosting, LOOCV
    python -m shadowrating.cli phase4         # combined divergence table + market-spread cross-check
    python -m shadowrating.cli phase5         # DSA scenarios + Monte Carlo debt-path fan charts
    python -m shadowrating.cli phase6         # export docs/data/*.json for the Chart.js dashboard

Phase 0 is the foundation: load the target, pull every available source, cache
each to data/raw/*.parquet, assemble the panel, and print an honest coverage
report. It never silently drops a country.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config, ratings as ratings_mod
from .loaders import worldbank, imf_weo, fred
from . import coverage as coverage_mod
from . import features as features_mod
from . import scorecard as scorecard_mod
from . import model_b as model_b_mod
from . import validation as validation_mod
from . import dsa as dsa_mod
from . import export_dashboard

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 60)


def cmd_ratings(_args) -> None:
    df = ratings_mod.load_ratings()
    cols = ["iso3", "name", "sp", "fitch", "moodys",
            "consensus_notch", "consensus_letter", "ig_flag", "split", "n_agencies"]
    print(df[cols].to_string(index=False))
    n_ig = int(df["ig_flag"].sum())
    print(f"\n{len(df)} sovereigns | {n_ig} investment grade | "
          f"{len(df) - n_ig} high yield | {int((df['split'] > 0).sum())} split-rated")


def cmd_wdi(args) -> None:
    d = worldbank.fetch_wdi(use_cache=not args.fresh)
    _summarize("WDI", d)


def cmd_wgi(args) -> None:
    d = worldbank.fetch_wgi(use_cache=not args.fresh)
    _summarize("WGI", d)


def cmd_weo(args) -> None:
    d = imf_weo.fetch_weo(use_cache=not args.fresh)
    _summarize("WEO", d)


def cmd_fred(args) -> None:
    d = fred.fetch_yields(use_cache=not args.fresh)
    if d.empty:
        print("[fred] no data (missing key or network).")
    else:
        print(d.to_string(index=False))


def cmd_phase0(args) -> None:
    fresh = args.fresh
    print(">>> Phase 0: target ratings + data loaders + coverage\n")

    rt = ratings_mod.load_ratings()
    print(f"[ratings] loaded {len(rt)} sovereigns "
          f"({int(rt['ig_flag'].sum())} IG, {len(rt) - int(rt['ig_flag'].sum())} HY)")

    frames = []
    for label, fn in [("WDI", lambda: worldbank.fetch_wdi(not fresh)),
                      ("WGI", lambda: worldbank.fetch_wgi(not fresh)),
                      ("WEO", lambda: imf_weo.fetch_weo(not fresh))]:
        try:
            d = fn()
            frames.append(d)
            _summarize(label, d)
        except RuntimeError as e:
            print(f"[{label.lower()}] SKIPPED: {e}")

    # FRED is validation, not a model feature -- pull but keep separate.
    try:
        yields = fred.fetch_yields(not fresh)
        if not yields.empty:
            print(f"[fred] {len(yields)} yields, benchmark = {config.FRED_BENCHMARK}")
    except RuntimeError as e:
        print(f"[fred] SKIPPED: {e}")

    panel = coverage_mod.assemble_panel(*frames)
    coverage_mod.print_coverage(panel)

    out = config.RAW_DIR / "panel.parquet"
    panel.to_parquet(out)
    print(f"Panel cached -> {out.relative_to(config.PROJECT_ROOT)}")
    print("Phase 0 complete. Next: Phase 1 (scaling + pillar features).")


def cmd_phase1(args) -> None:
    panel_path = config.RAW_DIR / "panel.parquet"
    if args.fresh or not panel_path.exists():
        cmd_phase0(args)
    panel = pd.read_parquet(panel_path)

    print("\n>>> Phase 1: scaling + pillar features\n")
    out = features_mod.build_features(panel)

    if out["empty_pillars"]:
        print(f"Pillars with no data this run (excluded, not zero-filled): "
              f"{', '.join(out['empty_pillars'])}")

    miss = out["missing_counts"]
    flagged = miss[(miss > 0).any(axis=1)]
    if not flagged.empty:
        print(f"\n{len(flagged)} countries had at least one missing indicator "
              f"within a pillar (pillar score = mean of what's available):")
        print(flagged[(flagged > 0).any(axis=1)].to_string())

    print("\nPillar scores (percentile rank, signed, [0, 1], higher = more creditworthy):")
    print(out["pillar_scores"].round(3).to_string())

    composite = out["composite"].sort_values(ascending=False)
    print("\nComposite score, ranked:")
    print(composite.round(3).to_string())

    processed_dir = config.DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out["pillar_scores"].to_parquet(processed_dir / "pillar_scores.parquet")
    out["composite"].to_frame("composite").to_parquet(processed_dir / "composite_score.parquet")
    print(f"\nFeatures cached -> {processed_dir.relative_to(config.PROJECT_ROOT)}/")
    print("Phase 1 complete. Next: Phase 2 (Model A rules-based scorecard).")


def cmd_phase2(args) -> None:
    panel_path = config.RAW_DIR / "panel.parquet"
    if args.fresh or not panel_path.exists():
        cmd_phase0(args)
    panel = pd.read_parquet(panel_path)
    feats = features_mod.build_features(panel)

    print("\n>>> Phase 2: Model A rules-based scorecard\n")
    rt = ratings_mod.load_ratings()
    table = scorecard_mod.build_scorecard(feats["pillar_scores"], feats["composite"], rt)

    calib = table.attrs["calibration"]
    print(f"In-sample calibration: notch = {calib['intercept']:.2f} + "
          f"{calib['slope']:.2f} * composite  (unit conversion only -- pillar "
          f"weights are fixed, not fitted)")

    cols = ["name", "composite", "loocv_pred_notch", "loocv_letter",
            "actual_notch", "actual_letter", "divergence"]
    print("\nPer-country scorecard, sorted by divergence (model vs. agencies, LOOCV):")
    print(table[cols].sort_values("divergence").round(2).to_string())

    summary = scorecard_mod.error_summary(table)
    print(f"\nLOOCV error (n={summary['n']}): MAE={summary['mae']:.2f} notches, "
          f"RMSE={summary['rmse']:.2f}, bias={summary['bias']:+.2f}")
    print(f"IG/HY boundary mismatches: {summary['ig_mismatch_count']}")
    if summary["ig_mismatch_count"]:
        print(summary["ig_mismatches"][cols].round(2).to_string())

    processed_dir = config.DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(processed_dir / "scorecard.parquet")
    print(f"\nScorecard cached -> {processed_dir.relative_to(config.PROJECT_ROOT)}/scorecard.parquet")
    print("Phase 2 complete. Next: Phase 3 (Model B -- ordered logit + gradient boosting, LOOCV).")


def cmd_phase3(args) -> None:
    panel_path = config.RAW_DIR / "panel.parquet"
    if args.fresh or not panel_path.exists():
        cmd_phase0(args)
    panel = pd.read_parquet(panel_path)
    feats = features_mod.build_features(panel)
    pillar_scores = feats["pillar_scores"]

    print("\n>>> Phase 3: Model B -- ordered logit + gradient boosting (LOOCV)\n")
    print(f"Features (pillar scores): {list(pillar_scores.columns)}")

    rt = ratings_mod.load_ratings()
    rt_idx = rt.set_index("iso3")
    notch = rt_idx["consensus_notch"].astype(float).reindex(pillar_scores.index)

    band_table = model_b_mod.fit_ordered_logit_loocv(pillar_scores, notch)
    band_table["name"] = rt_idx["name"].reindex(band_table.index)
    band_table["band_true_label"] = band_table["band_true"].map(ratings_mod.band_label)
    band_table["band_pred_label"] = band_table["band_pred"].round().map(ratings_mod.band_label)

    band_summary = model_b_mod.band_error_summary(band_table)
    print(f"Ordered logit LOOCV (n={band_summary['n']}): "
          f"exact band match={band_summary['exact_match_rate']:.0%}, "
          f"within one band={band_summary['within_one_band_rate']:.0%}, "
          f"MAE={band_summary['mae_bands']:.2f} bands")

    mismatches = band_table.dropna(subset=["band_true", "band_pred"])
    mismatches = mismatches[mismatches["band_true"] != mismatches["band_pred"].round()]
    if not mismatches.empty:
        print(f"\n{len(mismatches)} band mismatches:")
        print(mismatches[["name", "band_true_label", "band_pred_label"]].to_string())

    gbm_preds, importance = model_b_mod.fit_gbm_loocv(pillar_scores, notch)
    gbm_summary = model_b_mod.notch_error_summary(notch, gbm_preds)
    print(f"\nGradient boosting LOOCV (n={gbm_summary['n']}): "
          f"MAE={gbm_summary['mae']:.2f} notches, RMSE={gbm_summary['rmse']:.2f}, "
          f"bias={gbm_summary['bias']:+.2f}")
    print("\nMean feature importance across LOOCV folds (descriptive only, N too small "
          "for this to be a predictive ranking):")
    print(importance.round(3).to_string())

    print("\nFor comparison, Model A (Phase 2) LOOCV: see `phase2` output "
          "(MAE/RMSE on the same notch scale).")

    processed_dir = config.DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    band_table.to_parquet(processed_dir / "model_b_ordered_logit.parquet")
    gbm_preds.to_frame("gbm_pred_notch").to_parquet(processed_dir / "model_b_gbm.parquet")
    print(f"\nModel B outputs cached -> {processed_dir.relative_to(config.PROJECT_ROOT)}/")
    print("Phase 3 complete. Next: Phase 4 (validation + divergence table + market-spread cross-check).")


def cmd_phase4(args) -> None:
    panel_path = config.RAW_DIR / "panel.parquet"
    if args.fresh or not panel_path.exists():
        cmd_phase0(args)
    panel = pd.read_parquet(panel_path)
    feats = features_mod.build_features(panel)
    pillar_scores = feats["pillar_scores"]

    print("\n>>> Phase 4: validation -- combined divergence + market-spread cross-check\n")

    rt = ratings_mod.load_ratings()
    rt_idx = rt.set_index("iso3")
    notch = rt_idx["consensus_notch"].astype(float).reindex(pillar_scores.index)

    scorecard_table = scorecard_mod.build_scorecard(pillar_scores, feats["composite"], rt)
    band_table = model_b_mod.fit_ordered_logit_loocv(pillar_scores, notch)
    gbm_preds, _ = model_b_mod.fit_gbm_loocv(pillar_scores, notch)

    master = validation_mod.build_master_table(scorecard_table, band_table, gbm_preds, rt)

    cols = ["name", "actual_letter", "model_a_divergence", "model_b_gbm_divergence",
            "models_agree_direction"]
    print("Combined divergence table, sorted by max |divergence| across both models:")
    print(master.sort_values("max_abs_divergence", ascending=False)[cols].round(2).to_string())

    agree = master[master["models_agree_direction"]]
    print(f"\n{len(agree)} countries where Model A and Model B's gradient-boosted "
          f"regressor diverge from the agencies in the *same* direction "
          f"(the strongest version of the signal -- two differently-built "
          f"models, one consistent disagreement with the agencies):")
    if not agree.empty:
        print(agree.sort_values("max_abs_divergence", ascending=False)[cols].round(2).to_string())

    try:
        yields = fred.fetch_yields(not args.fresh)
    except RuntimeError as e:
        yields = pd.DataFrame()
        print(f"\n[fred] SKIPPED: {e}")

    cross = validation_mod.cross_check_market_spread(master, yields)
    if cross is None:
        print("\n[market cross-check] SKIPPED -- no FRED data this run "
              "(set FRED_API_KEY and rerun to enable). Coverage is OECD-only "
              "even when enabled, so treat this as a sanity check on "
              "rich-country pricing, not full validation.")
    else:
        print("\nMarket-spread cross-check (FRED-covered subset only; spread "
              "residual = actual spread minus what the agency notch alone "
              "would predict):")
        print(cross.round(2).to_string())
        n_corrob = int(cross["model_a_corroborated"].sum())
        print(f"\nModel A's divergence direction is corroborated by the market "
              f"in {n_corrob}/{len(cross)} FRED-covered cases.")

    processed_dir = config.DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    master.to_parquet(processed_dir / "validation_master.parquet")
    print(f"\nValidation table cached -> {processed_dir.relative_to(config.PROJECT_ROOT)}/validation_master.parquet")
    print("Phase 4 complete. Next: Phase 5 (DSA scenario module + debt-path fan charts).")


def cmd_phase5(args) -> None:
    panel_path = config.RAW_DIR / "panel.parquet"
    if args.fresh or not panel_path.exists():
        cmd_phase0(args)
    panel = pd.read_parquet(panel_path)

    print("\n>>> Phase 5: DSA scenarios + Monte Carlo debt-path fan charts\n")
    print(f"Horizon: {dsa_mod.HORIZON_YEARS} years. Stress shocks (illustrative, "
          f"not estimated from this country's own historical volatility -- see "
          f"shadowrating/dsa.py docstring): growth {dsa_mod.GROWTH_SHOCK_PP:+.1f}pp, "
          f"rate {dsa_mod.RATE_SHOCK_PP:+.1f}pp, primary balance "
          f"{dsa_mod.PRIMARY_BALANCE_SHOCK_PP:+.1f}pp of GDP.")

    inputs = dsa_mod.dsa_inputs(panel)
    dropped = set(panel.index) - set(inputs.index)
    if dropped:
        print(f"\n{len(dropped)} countries dropped (missing fiscal/growth data, "
              f"not imputed): {sorted(dropped)}")

    scenarios_long, fan_long = dsa_mod.build_dsa_tables(panel)
    frag = dsa_mod.fragility_summary(scenarios_long)
    rt = ratings_mod.load_ratings().set_index("iso3")
    frag["name"] = rt["name"].reindex(frag.index)
    frag["actual_letter"] = rt["consensus_letter"].reindex(frag.index)

    print(f"\nImplied effective interest rate vs. nominal growth (r > g countries "
          f"see baseline debt/GDP rise even with no shock):")
    rg = inputs[["i0", "nominal_g0"]].copy()
    rg["r_minus_g"] = rg["i0"] - rg["nominal_g0"]
    rg["name"] = rt["name"].reindex(rg.index)
    print(rg.sort_values("r_minus_g", ascending=False).head(10).round(2).to_string())

    print(f"\nMost fragile (largest combined-adverse vs. baseline debt/GDP gap at "
          f"year {dsa_mod.HORIZON_YEARS}):")
    cols = ["name", "actual_letter", "debt_gdp_now", "baseline_end",
            "combined_adverse_end", "baseline_rising"]
    print(frag.sort_values("adverse_minus_baseline", ascending=False).head(10)[cols].round(1).to_string())

    n_rising = int(frag["baseline_rising"].sum())
    print(f"\n{n_rising}/{len(frag)} countries have debt/GDP still rising under "
          f"baseline (no shocks) after {dsa_mod.HORIZON_YEARS} years.")

    processed_dir = config.DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    scenarios_long.to_parquet(processed_dir / "dsa_scenarios.parquet")
    fan_long.to_parquet(processed_dir / "dsa_fan.parquet")
    frag.to_parquet(processed_dir / "dsa_fragility.parquet")
    print(f"\nDSA outputs cached -> {processed_dir.relative_to(config.PROJECT_ROOT)}/ "
          f"(dsa_scenarios, dsa_fan, dsa_fragility) -- ready for the Phase 6 dashboard.")
    print("Phase 5 complete. Next: Phase 6 (Chart.js dashboard, host on GitHub Pages).")


def cmd_phase6(args) -> None:
    print("\n>>> Phase 6: exporting dashboard data\n")
    export_dashboard.build_all(use_cache=not args.fresh)
    print("\nPhase 6 export complete. To view locally:\n"
          "  cd docs && python -m http.server 8000\n"
          "  open http://localhost:8000\n"
          "(must be served over http://, not opened as a file:// path -- "
          "fetch() for the JSON won't work from disk.)")


def _summarize(label: str, d: pd.DataFrame) -> None:
    if d is None or d.empty:
        print(f"[{label.lower()}] no rows returned.")
        return
    print(f"[{label.lower()}] {len(d)} rows | {d['iso3'].nunique()} countries | "
          f"{d['indicator'].nunique()} indicators")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shadowrating", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    for name, fn in [("ratings", cmd_ratings), ("wdi", cmd_wdi), ("wgi", cmd_wgi),
                     ("weo", cmd_weo), ("fred", cmd_fred), ("phase0", cmd_phase0),
                     ("phase1", cmd_phase1), ("phase2", cmd_phase2), ("phase3", cmd_phase3),
                     ("phase4", cmd_phase4), ("phase5", cmd_phase5), ("phase6", cmd_phase6)]:
        sp = sub.add_parser(name, help=fn.__doc__)
        sp.add_argument("--fresh", action="store_true",
                        help="ignore caches and re-pull from source")
        sp.set_defaults(func=fn)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
