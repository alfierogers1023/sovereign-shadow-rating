"""
Assemble the indicator sources into one country x indicator panel and report
coverage honestly. Nothing is imputed or dropped here -- this module's whole
job is to make the gaps visible (the spec is emphatic about never silently
dropping the distressed countries, which is exactly where coverage fails).
"""
from __future__ import annotations

import pandas as pd

from . import config


def _dedup_long(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concat tidy frames and keep the latest-year observation per (iso3, indicator)."""
    nonempty = [f for f in frames if f is not None and not f.empty]
    if not nonempty:
        return pd.DataFrame(columns=["iso3", "indicator", "year", "value"])
    long = pd.concat(nonempty, ignore_index=True)
    long = long.sort_values("year", na_position="first")
    return long.groupby(["iso3", "indicator"], as_index=False).last()


def assemble_panel(*frames: pd.DataFrame) -> pd.DataFrame:
    """
    Combine tidy (iso3, indicator, year, value) frames into a wide panel:
    one row per country, one column per indicator (most-recent value).
    """
    long = _dedup_long(*frames)
    if long.empty:
        return pd.DataFrame(index=pd.Index(config.ISO3_LIST, name="iso3"))

    wide = long.pivot(index="iso3", columns="indicator", values="value")
    # Reindex to the full sample so missing countries show as all-NaN rows.
    wide = wide.reindex(config.ISO3_LIST)
    wide.index.name = "iso3"
    return wide


def assemble_vintage(*frames: pd.DataFrame) -> pd.DataFrame:
    """
    Same assembly as `assemble_panel`, but pivots the vintage `year` instead
    of `value` -- one column per indicator giving the year the value actually
    used (after taking the latest available per country/indicator) was
    published for. Lets a reader see how stale any given country/pillar's
    inputs are, which `assemble_panel` alone throws away.
    """
    long = _dedup_long(*frames)
    if long.empty:
        return pd.DataFrame(index=pd.Index(config.ISO3_LIST, name="iso3"))

    wide = long.pivot(index="iso3", columns="indicator", values="year")
    wide = wide.reindex(config.ISO3_LIST)
    wide.index.name = "iso3"
    return wide


def coverage_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Boolean present/absent matrix (countries x indicators)."""
    return panel.notna()


def coverage_report(panel: pd.DataFrame) -> dict:
    """Summary stats + the worst-covered countries and indicators."""
    present = panel.notna()
    by_country = present.mean(axis=1).sort_values()
    by_indicator = present.mean(axis=0).sort_values()
    overall = present.values.mean() if present.size else 0.0

    drop_candidates = by_indicator[by_indicator < config.COVERAGE_DROP_THRESHOLD]

    return {
        "overall_fill_rate": overall,
        "by_country": by_country,
        "by_indicator": by_indicator,
        "drop_candidates": list(drop_candidates.index),
    }


def print_coverage(panel: pd.DataFrame) -> None:
    rep = coverage_report(panel)
    print("\n=== COVERAGE REPORT =======================================")
    print(f"Panel: {panel.shape[0]} countries x {panel.shape[1]} indicators")
    print(f"Overall fill rate: {rep['overall_fill_rate']:.1%}")

    print("\nLeast-covered countries (fill rate):")
    worst_c = rep["by_country"].head(8)
    for iso3, rate in worst_c.items():
        name = config.COUNTRIES.get(iso3, {}).get("name", iso3)
        print(f"  {iso3} {name:<18} {rate:5.0%}")

    print("\nLeast-covered indicators (fill rate):")
    worst_i = rep["by_indicator"].head(8)
    for ind, rate in worst_i.items():
        print(f"  {ind:<28} {rate:5.0%}")

    if rep["drop_candidates"]:
        print(f"\nBelow {config.COVERAGE_DROP_THRESHOLD:.0%} threshold "
              f"(consider dropping or documenting imputation):")
        print("  " + ", ".join(rep["drop_candidates"]))
    else:
        print(f"\nAll indicators meet the {config.COVERAGE_DROP_THRESHOLD:.0%} "
              "coverage threshold.")
    print("===========================================================\n")
