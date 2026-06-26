"""
Phase 5: IMF-style debt sustainability analysis (DSA).

Standard debt dynamics identity (nominal terms, debt and primary balance as
% of GDP):

    d_t = d_{t-1} * (1 + i_t) / (1 + g_t) - pb_t

where i = effective nominal interest rate paid on the debt stock, g = nominal
GDP growth, pb = primary balance (positive = surplus, reduces debt).

Inputs come from the *raw* WEO panel (NOT the percentile-scaled features from
Phase 1 -- this module needs actual percent-of-GDP levels, not ranks):
gov_gross_debt_gdp, primary_balance_gdp, gov_net_lending_gdp,
real_gdp_growth_weo, inflation_weo.

The effective interest rate isn't published directly -- it's backed out from
the accounting identity overall_balance = primary_balance - interest:

    interest_bill_pct_gdp = primary_balance_gdp - gov_net_lending_gdp
    i = interest_bill_pct_gdp / (debt_gdp / 100)

This is a derived *average* effective rate on the existing stock, not a
market-observed marginal rate -- treat it as an approximation, most reliable
for sovereigns with normal (non-distressed, non-multi-currency) debt
structures.

Stress-scenario shock sizes (-1.5pp growth, +200bp on the effective rate,
-1pp of GDP on the primary balance) are illustrative IMF-DSA-style
magnitudes, not estimated from this country's own historical volatility --
we only have one WEO vintage per indicator here, not a time series to
estimate volatility from. Said plainly here and in CLI output, not quietly
assumed away. Same caveat applies to the Monte-Carlo fan chart's shock
standard deviations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_YEARS = 10

# Illustrative shock magnitudes -- see module docstring caveat.
GROWTH_SHOCK_PP = -1.5
RATE_SHOCK_PP = 2.0
PRIMARY_BALANCE_SHOCK_PP = -1.0

# Monte Carlo fan-chart shock std devs (also illustrative, same caveat).
SIGMA_GROWTH_PP = 1.5
SIGMA_RATE_PP = 1.0
SIGMA_PRIMARY_BALANCE_PP = 1.0
N_SIMULATIONS = 2000
FAN_PERCENTILES = [10, 25, 50, 75, 90]

DSA_PANEL_COLS = ["gov_gross_debt_gdp", "primary_balance_gdp",
                  "gov_net_lending_gdp", "real_gdp_growth_weo", "inflation_weo"]


def nominal_growth_pct(real_growth_pct: pd.Series, inflation_pct: pd.Series) -> pd.Series:
    return ((1 + real_growth_pct / 100) * (1 + inflation_pct / 100) - 1) * 100


def dsa_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One row per country: starting debt/GDP, primary balance, nominal growth,
    and the implied effective interest rate -- everything project_path needs.
    Countries missing any of the five raw WEO columns are dropped (not
    imputed -- a DSA built on a guessed debt level or interest rate isn't
    honest).
    """
    sub = panel[DSA_PANEL_COLS].dropna()
    out = pd.DataFrame(index=sub.index)
    out["debt_gdp0"] = sub["gov_gross_debt_gdp"]
    out["pb0"] = sub["primary_balance_gdp"]
    out["nominal_g0"] = nominal_growth_pct(sub["real_gdp_growth_weo"], sub["inflation_weo"])
    interest_bill = sub["primary_balance_gdp"] - sub["gov_net_lending_gdp"]
    out["i0"] = interest_bill / (sub["gov_gross_debt_gdp"] / 100)
    return out


def project_path(d0: float, i_pct: float, g_pct: float, pb_pct: float,
                 horizon: int = HORIZON_YEARS) -> np.ndarray:
    """Deterministic debt/GDP path, constant rate/growth/primary balance throughout."""
    path = np.empty(horizon + 1)
    path[0] = d0
    i, g = i_pct / 100, g_pct / 100
    for t in range(1, horizon + 1):
        path[t] = path[t - 1] * (1 + i) / (1 + g) - pb_pct
    return path


def build_scenarios(row: pd.Series, horizon: int = HORIZON_YEARS) -> dict[str, np.ndarray]:
    """Baseline + three single-factor stress tests + one combined adverse scenario."""
    d0, pb0, g0, i0 = row["debt_gdp0"], row["pb0"], row["nominal_g0"], row["i0"]
    return {
        "baseline": project_path(d0, i0, g0, pb0, horizon),
        "growth_shock": project_path(d0, i0, g0 + GROWTH_SHOCK_PP, pb0, horizon),
        "rate_shock": project_path(d0, i0 + RATE_SHOCK_PP, g0, pb0, horizon),
        "fiscal_shock": project_path(d0, i0, g0, pb0 + PRIMARY_BALANCE_SHOCK_PP, horizon),
        "combined_adverse": project_path(
            d0, i0 + RATE_SHOCK_PP, g0 + GROWTH_SHOCK_PP,
            pb0 + PRIMARY_BALANCE_SHOCK_PP, horizon,
        ),
    }


def simulate_fan(row: pd.Series, horizon: int = HORIZON_YEARS,
                 n_sims: int = N_SIMULATIONS, seed: int = 0) -> pd.DataFrame:
    """
    Monte Carlo fan chart: each simulated path draws an iid shock to growth,
    rate, and primary balance every year (independent across years and
    paths, centred on the baseline). Returns year x percentile debt/GDP.
    """
    rng = np.random.default_rng(seed)
    d0, pb0, g0, i0 = row["debt_gdp0"], row["pb0"], row["nominal_g0"], row["i0"]

    paths = np.empty((n_sims, horizon + 1))
    paths[:, 0] = d0
    g_draws = rng.normal(g0, SIGMA_GROWTH_PP, size=(n_sims, horizon))
    i_draws = rng.normal(i0, SIGMA_RATE_PP, size=(n_sims, horizon))
    pb_draws = rng.normal(pb0, SIGMA_PRIMARY_BALANCE_PP, size=(n_sims, horizon))

    for t in range(1, horizon + 1):
        i_t, g_t = i_draws[:, t - 1] / 100, g_draws[:, t - 1] / 100
        paths[:, t] = paths[:, t - 1] * (1 + i_t) / (1 + g_t) - pb_draws[:, t - 1]

    pct = np.percentile(paths, FAN_PERCENTILES, axis=0)
    return pd.DataFrame(pct.T, columns=[f"p{p}" for p in FAN_PERCENTILES],
                        index=pd.RangeIndex(horizon + 1, name="year_ahead"))


def build_dsa_tables(panel: pd.DataFrame, horizon: int = HORIZON_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (scenarios_long, fan_long):
      scenarios_long: iso3 | year_ahead | scenario | debt_gdp
      fan_long:       iso3 | year_ahead | p10..p90 (Monte Carlo percentiles)
    """
    inputs = dsa_inputs(panel)

    scenario_rows = []
    fan_rows = []
    for iso3, row in inputs.iterrows():
        scenarios = build_scenarios(row, horizon)
        for name, path in scenarios.items():
            for year_ahead, value in enumerate(path):
                scenario_rows.append({"iso3": iso3, "year_ahead": year_ahead,
                                      "scenario": name, "debt_gdp": value})

        fan = simulate_fan(row, horizon)
        fan = fan.reset_index()
        fan["iso3"] = iso3
        fan_rows.append(fan)

    scenarios_long = pd.DataFrame(scenario_rows)
    fan_long = pd.concat(fan_rows, ignore_index=True)
    return scenarios_long, fan_long


def fragility_summary(scenarios_long: pd.DataFrame, horizon: int = HORIZON_YEARS) -> pd.DataFrame:
    """
    One row per country: starting debt/GDP, baseline and combined-adverse
    debt/GDP at the end of the horizon, and whether debt is still rising
    under baseline at that point -- the classic DSA "sustainability" red flag.
    """
    end = scenarios_long[scenarios_long["year_ahead"] == horizon]
    wide = end.pivot(index="iso3", columns="scenario", values="debt_gdp")
    start = scenarios_long[scenarios_long["year_ahead"] == 0].drop_duplicates("iso3").set_index("iso3")["debt_gdp"]

    out = pd.DataFrame(index=wide.index)
    out["debt_gdp_now"] = start
    out["baseline_end"] = wide["baseline"]
    out["combined_adverse_end"] = wide["combined_adverse"]
    out["baseline_rising"] = out["baseline_end"] > out["debt_gdp_now"]
    out["adverse_minus_baseline"] = out["combined_adverse_end"] - out["baseline_end"]
    return out
