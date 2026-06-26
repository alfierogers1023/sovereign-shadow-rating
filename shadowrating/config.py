"""
Central configuration for the sovereign shadow-rating model.

Everything that drifts (country sample, indicator codes, data vintage) lives here
so the rest of the codebase never hard-codes a magic string. Verify indicator
codes at build time -- World Bank / IMF codes and coverage do change.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # cached API pulls (parquet) + manual downloads
RATINGS_CSV = DATA_DIR / "ratings.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Country sample (~40), spanning the full rating scale.
# Keys are ISO-3166 alpha-3 codes (what World Bank + IMF WEO use).
# `bucket` is just a coarse label for sanity-checking coverage across the scale.
# --------------------------------------------------------------------------- #
COUNTRIES: dict[str, dict[str, str]] = {
    # AAA / AA
    "USA": {"name": "United States", "bucket": "AAA/AA"},
    "DEU": {"name": "Germany", "bucket": "AAA/AA"},
    "NLD": {"name": "Netherlands", "bucket": "AAA/AA"},
    "CAN": {"name": "Canada", "bucket": "AAA/AA"},
    "AUS": {"name": "Australia", "bucket": "AAA/AA"},
    "SWE": {"name": "Sweden", "bucket": "AAA/AA"},
    "CHE": {"name": "Switzerland", "bucket": "AAA/AA"},
    "GBR": {"name": "United Kingdom", "bucket": "AAA/AA"},
    "FRA": {"name": "France", "bucket": "AAA/AA"},
    "JPN": {"name": "Japan", "bucket": "AAA/AA"},
    # A
    "KOR": {"name": "South Korea", "bucket": "A"},
    "BEL": {"name": "Belgium", "bucket": "A"},
    "CZE": {"name": "Czechia", "bucket": "A"},
    "CHL": {"name": "Chile", "bucket": "A"},
    "SAU": {"name": "Saudi Arabia", "bucket": "A"},
    "CHN": {"name": "China", "bucket": "A"},
    "POL": {"name": "Poland", "bucket": "A"},
    # BBB
    "ITA": {"name": "Italy", "bucket": "BBB"},
    "ESP": {"name": "Spain", "bucket": "BBB"},
    "PRT": {"name": "Portugal", "bucket": "BBB"},
    "MEX": {"name": "Mexico", "bucket": "BBB"},
    "IDN": {"name": "Indonesia", "bucket": "BBB"},
    "IND": {"name": "India", "bucket": "BBB"},
    "PHL": {"name": "Philippines", "bucket": "BBB"},
    "HUN": {"name": "Hungary", "bucket": "BBB"},
    "ROU": {"name": "Romania", "bucket": "BBB"},
    # BB
    "BRA": {"name": "Brazil", "bucket": "BB"},
    "ZAF": {"name": "South Africa", "bucket": "BB"},
    "COL": {"name": "Colombia", "bucket": "BB"},
    "GRC": {"name": "Greece", "bucket": "BB"},
    "TUR": {"name": "Turkey", "bucket": "BB"},
    "MAR": {"name": "Morocco", "bucket": "BB"},
    # B
    "EGY": {"name": "Egypt", "bucket": "B"},
    "NGA": {"name": "Nigeria", "bucket": "B"},
    "KEN": {"name": "Kenya", "bucket": "B"},
    "PAK": {"name": "Pakistan", "bucket": "B"},
    "AGO": {"name": "Angola", "bucket": "B"},
    # CCC and below
    "ARG": {"name": "Argentina", "bucket": "CCC-"},
    "UKR": {"name": "Ukraine", "bucket": "CCC-"},
    "ZMB": {"name": "Zambia", "bucket": "CCC-"},
    "GHA": {"name": "Ghana", "bucket": "CCC-"},
    "LKA": {"name": "Sri Lanka", "bucket": "CCC-"},
}

ISO3_LIST = list(COUNTRIES.keys())

# --------------------------------------------------------------------------- #
# Indicator panel.
# Maps source code -> short friendly name + direction.
# direction = +1 means "higher value is better for creditworthiness",
#            -1 means "higher value is worse" (debt, inflation, etc.) -> flip when scaling.
# --------------------------------------------------------------------------- #

# World Bank WDI (database 2 in wbgapi)
WDI_INDICATORS: dict[str, dict] = {
    "NY.GDP.PCAP.PP.CD":  {"name": "gdp_per_capita_ppp", "pillar": "economic", "direction": +1},
    "NY.GDP.MKTP.KD.ZG":  {"name": "real_gdp_growth",    "pillar": "economic", "direction": +1},
    "NY.GDP.MKTP.CD":     {"name": "gdp_usd",            "pillar": "economic", "direction": +1},
    "FP.CPI.TOTL.ZG":     {"name": "inflation_cpi",      "pillar": "monetary", "direction": -1},
    "FI.RES.TOTL.MO":     {"name": "reserves_months_imports", "pillar": "external", "direction": +1},
    "BN.CAB.XOKA.GD.ZS":  {"name": "current_account_gdp_wdi", "pillar": "external", "direction": +1},
}

# World Bank WGI (database 3 in wbgapi). Estimate series, range roughly [-2.5, +2.5].
# World Bank renamed these from bare codes (e.g. "GE.EST") to "GOV_WGI_"-prefixed
# codes; the bare codes now 404 against the live API.
WGI_INDICATORS: dict[str, dict] = {
    "GOV_WGI_GE.EST": {"name": "gov_effectiveness",   "pillar": "institutional", "direction": +1},
    "GOV_WGI_RL.EST": {"name": "rule_of_law",         "pillar": "institutional", "direction": +1},
    "GOV_WGI_CC.EST": {"name": "control_corruption",  "pillar": "institutional", "direction": +1},
    "GOV_WGI_PV.EST": {"name": "political_stability",  "pillar": "institutional", "direction": +1},
    "GOV_WGI_RQ.EST": {"name": "regulatory_quality",  "pillar": "institutional", "direction": +1},
    "GOV_WGI_VA.EST": {"name": "voice_accountability", "pillar": "institutional", "direction": +1},
}

# IMF WEO subject codes. These double as both the WEO flat-file "Subject
# Code" values and IMF DataMapper API indicator codes -- the two sources use
# matching codes except primary balance (DataMapper: GGXONLB_G01_GDP_PT vs.
# flat-file: GGXONLB_NGDP), handled in loaders/imf_weo.py.
WEO_INDICATORS: dict[str, dict] = {
    "GGXWDG_NGDP":  {"name": "gov_gross_debt_gdp",   "pillar": "fiscal",   "direction": -1},
    "GGXCNL_NGDP":  {"name": "gov_net_lending_gdp",  "pillar": "fiscal",   "direction": +1},
    "GGXONLB_NGDP": {"name": "primary_balance_gdp",  "pillar": "fiscal",   "direction": +1},
    "BCA_NGDPD":    {"name": "current_account_gdp_weo", "pillar": "external", "direction": +1},
    "NGDP_RPCH":    {"name": "real_gdp_growth_weo",   "pillar": "economic", "direction": +1},
    "PCPIPCH":      {"name": "inflation_weo",         "pillar": "monetary", "direction": -1},
}

# Flat-file subject code -> DataMapper API indicator code, for the one
# indicator where IMF's two WEO distribution channels disagree on naming.
WEO_DATAMAPPER_CODE_OVERRIDES: dict[str, str] = {
    "GGXONLB_NGDP": "GGXONLB_G01_GDP_PT",
}

# Pillar weights for Model A (illustrative starting point; justify against a named
# agency methodology in the write-up before using these for anything real).
PILLAR_WEIGHTS: dict[str, float] = {
    "economic": 0.25,
    "institutional": 0.20,
    "fiscal": 0.25,
    "external": 0.15,
    "monetary": 0.15,
}

# --------------------------------------------------------------------------- #
# FRED long-term government bond yields (validation / early-warning).
# Series pattern: IRLTLT01{CC}M156N  (monthly, % per annum). OECD coverage only.
# CC = OECD 2-letter code. EM names mostly absent -> fall back to spread data later.
# --------------------------------------------------------------------------- #
FRED_YIELD_PATTERN = "IRLTLT01{cc}M156N"
FRED_BENCHMARK = "DEU"   # compute spread vs this country's yield (Germany)

# iso3 -> FRED 2-letter country code, for the subset FRED covers.
FRED_COUNTRY_CODES: dict[str, str] = {
    "USA": "US", "DEU": "DE", "NLD": "NL", "CAN": "CA", "AUS": "AU",
    "SWE": "SE", "CHE": "CH", "GBR": "GB", "FRA": "FR", "JPN": "JP",
    "KOR": "KR", "BEL": "BE", "CZE": "CZ", "CHL": "CL", "POL": "PL",
    "ITA": "IT", "ESP": "ES", "PRT": "PT", "MEX": "MX", "IDN": "ID",
    "IND": "IN", "HUN": "HU", "COL": "CO", "GRC": "GR", "TUR": "TR",
}

# --------------------------------------------------------------------------- #
# Data vintage. None -> take most-recent available value per indicator per country.
# Set to an int (e.g. 2024) to pin a fixed vintage year instead.
# --------------------------------------------------------------------------- #
VINTAGE_YEAR: int | None = None
COVERAGE_DROP_THRESHOLD = 0.80   # drop an indicator if coverage < this share of countries


def all_indicator_names() -> list[str]:
    """Friendly names of every indicator across all three sources."""
    out = []
    for d in (WDI_INDICATORS, WGI_INDICATORS, WEO_INDICATORS):
        out.extend(v["name"] for v in d.values())
    return out
