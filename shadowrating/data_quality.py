"""
Signal vs. artefact: distinguish a real rating divergence from one driven by
stale or thin data, rather than a genuine disagreement with the agencies.

Two metrics, both straight from data the rest of the pipeline already
computes -- nothing new is fetched:

1. Vintage. WDI, WGI, and WEO each publish on a different lag, and not every
   country reports on the same cycle -- "most recent available" can mean very
   different things across countries. `coverage.assemble_vintage` keeps the
   year that survived `assemble_panel`'s "latest available" dedup; this module
   turns that into an age (years behind `as_of_year`) per country.

   Absolute age alone is a weak signal here: WDI and WGI both lag ~2 years
   for literally every country in the sample (that's just how those sources
   publish), so "max age across all indicators" is ~2 years for almost
   everyone and tells you nothing about which countries are unusual. What
   actually isolates a country-specific data gap is *excess* staleness: each
   indicator's age minus that same indicator's age for the median country.
   Concretely: every country's WEO fiscal data is 2026 (age 0) except Sri
   Lanka, stuck at 2024 (age 2, +2 years excess vs. every single peer) and one
   Zambia indicator (2025, +1 year). That is the real, country-specific
   finding -- not "Sri Lanka's data is 2 years old" (so is everyone's WDI/WGI
   data) but "Sri Lanka's *fiscal* pillar specifically is 2 years older than
   the sample norm," for a country already flagged as a large model-vs-agency
   divergence.

2. "Imputed" share. Nothing in this pipeline is ever filled with a guessed
   value (see features.py) -- a missing indicator just means a pillar score
   is the mean of fewer indicators than usual. We report the share of
   (country, pillar, indicator) slots that were missing as a proxy for "how
   much of this score rests on partial evidence," while being explicit that
   the word "imputed" overstates what actually happens: no number is
   invented, the average just has a smaller denominator. At 100% panel
   coverage (current state of this project), this is 0% for every country --
   worth reporting precisely *because* it's currently unremarkable; it exists
   to catch the day coverage degrades, especially for the distressed names
   coverage gaps cluster around.
"""
from __future__ import annotations

import datetime

import pandas as pd

from . import features as features_mod


def indicator_age(vintage_panel: pd.DataFrame, as_of_year: int | None = None) -> pd.DataFrame:
    """Country x indicator age in years: as_of_year - vintage year. NaN stays NaN."""
    as_of_year = as_of_year or datetime.date.today().year
    return as_of_year - vintage_panel


def excess_age(age: pd.DataFrame) -> pd.DataFrame:
    """
    Per indicator, age minus that indicator's age for the *median* country --
    isolates a country-specific data gap from a source's universal
    publication lag (which would otherwise swamp every other signal; see
    module docstring).
    """
    return age.sub(age.median(axis=0), axis=1)


def pillar_vintage_summary(vintage_panel: pd.DataFrame, as_of_year: int | None = None) -> pd.DataFrame:
    """
    Country x pillar: max absolute age and max excess age (vs. the sample
    median for each indicator) within that pillar. Excess age is the one
    that actually flags a country-specific gap rather than a source's
    universal lag -- see module docstring.
    """
    age = indicator_age(vintage_panel, as_of_year)
    excess = excess_age(age)
    meta = features_mod.indicator_meta()
    pillars = sorted(set(m["pillar"] for m in meta.values()))
    out_max, out_excess = {}, {}
    for p in pillars:
        cols = [c for c in age.columns if meta.get(c, {}).get("pillar") == p]
        if not cols:
            continue
        out_max[p] = age[cols].max(axis=1)
        out_excess[p] = excess[cols].max(axis=1)
    return pd.concat({"max_age": pd.DataFrame(out_max), "max_excess_age": pd.DataFrame(out_excess)}, axis=1)


def overall_vintage_summary(vintage_panel: pd.DataFrame, as_of_year: int | None = None) -> pd.DataFrame:
    """
    One row per country: oldest indicator used anywhere in absolute terms
    (max_age_years -- mostly reflects each source's universal lag, included
    for context), and the more diagnostic max_excess_age_years (how far this
    country's *most unusual* indicator sits above the sample norm for that
    same indicator) plus every pillar touched by a stale indicator
    (stale_pillars) -- not just the single worst one. A country whose entire
    WEO release is one vintage behind (Sri Lanka: fiscal, external, economic,
    *and* monetary all tied at +2 years) would be understated by "the worst
    pillar is X" -- report the full set so a reader sees the real exposure.
    """
    age = indicator_age(vintage_panel, as_of_year)
    excess = excess_age(age)
    meta = features_mod.indicator_meta()
    col_to_pillar = {c: meta.get(c, {}).get("pillar") for c in age.columns}

    out = pd.DataFrame(index=age.index)
    out["max_age_years"] = age.max(axis=1)
    out["max_excess_age_years"] = excess.max(axis=1)

    stale_pillars = pd.Series([[] for _ in age.index], index=age.index, dtype=object)
    for iso3 in age.index:
        row_max = out.loc[iso3, "max_excess_age_years"]
        if pd.isna(row_max) or row_max <= 0:
            continue
        stale_cols = excess.columns[excess.loc[iso3] == row_max]
        stale_pillars.loc[iso3] = sorted({col_to_pillar[c] for c in stale_cols})
    out["stale_pillars"] = stale_pillars
    out["worst_pillar"] = stale_pillars.map(lambda ps: ps[0] if ps else None)
    return out


def missing_share(missing_counts: pd.DataFrame, total_indicators: int) -> pd.Series:
    """
    Per country: share of (pillar, indicator) slots missing and therefore
    excluded from a pillar's average, out of every indicator slot that
    exists anywhere in this panel (empty pillars, e.g. one with zero
    indicators at all, don't count against anyone -- there's nothing to be
    "missing" from).
    """
    if total_indicators == 0:
        return pd.Series(0.0, index=missing_counts.index)
    return missing_counts.sum(axis=1) / total_indicators


def build_data_quality_table(vintage_panel: pd.DataFrame, missing_counts: pd.DataFrame,
                             total_indicators: int, as_of_year: int | None = None) -> pd.DataFrame:
    """One row per country: absolute + excess data age, worst pillar, and imputed share."""
    vintage = overall_vintage_summary(vintage_panel, as_of_year)
    vintage["missing_share"] = missing_share(missing_counts, total_indicators)
    return vintage
