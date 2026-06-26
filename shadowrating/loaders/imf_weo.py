"""
IMF World Economic Outlook (WEO) loader.

Two paths, tried in order:

1. A local flat file in data/raw/ (weo.tsv etc.) -- the fully reproducible,
   offline path. Get one: go to
   https://www.imf.org/en/Publications/WEO/weo-database, download "By
   Countries" -> the full "all" tab-delimited dataset (arrives as
   WEO<Month><Year>all.xls but is actually tab-separated text), and save it as
   data/raw/weo.tsv.
2. IMF's DataMapper API (https://www.imf.org/external/datamapper/api/v1/),
   used automatically if no flat file is present. It's IMF's own stable public
   JSON API for the same WEO release, not a scraped workaround -- but it is a
   live network call (cached like every other loader), so prefer the flat
   file if you want a fully offline, pinned-vintage run.

If neither is available, this loader prints download instructions and returns
an empty tidy frame rather than crashing the pipeline.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import requests

from .. import config
from . import TIDY_COLS, save_cache, load_cache

DATAMAPPER_BASE = "https://www.imf.org/external/datamapper/api/v1"

# Candidate filenames we'll look for in data/raw/.
_CANDIDATES = ["weo.tsv", "weo.txt", "weo.xls", "weo.csv"]


def _find_weo_file() -> Path | None:
    for name in _CANDIDATES:
        p = config.RAW_DIR / name
        if p.exists():
            return p
    # Also accept any WEO*all* file dropped in raw/ verbatim.
    for p in config.RAW_DIR.glob("WEO*all*"):
        return p
    return None


def _print_download_help() -> None:
    print(
        "[weo] No WEO flat file found in data/raw/.\n"
        "      Download the tab-delimited 'all' dataset from\n"
        "      https://www.imf.org/en/Publications/WEO/weo-database\n"
        "      and save it as data/raw/weo.tsv (tab-separated), then rerun."
    )


def _fetch_datamapper() -> pd.DataFrame:
    """
    Pull each WEO indicator from IMF's DataMapper API. Each call returns every
    country IMF tracks for that indicator, including forecast years out to
    ~2030 -- we cap at the current calendar year so this stays a "current
    state" snapshot, not a forward projection mixed in with actuals.
    """
    current_year = datetime.date.today().year
    rows = []
    for code, meta in config.WEO_INDICATORS.items():
        api_code = config.WEO_DATAMAPPER_CODE_OVERRIDES.get(code, code)
        url = f"{DATAMAPPER_BASE}/{api_code}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[weo] DataMapper call failed for {api_code}: {e}")
            continue

        by_country = payload.get("values", {}).get(api_code, {})
        for iso3 in config.ISO3_LIST:
            series = by_country.get(iso3)
            if not series:
                continue
            valid_years = [y for y in series if y.isdigit() and int(y) <= current_year
                           and series[y] is not None]
            if not valid_years:
                continue
            latest = max(valid_years, key=int)
            rows.append({"iso3": iso3, "indicator": meta["name"],
                        "year": int(latest), "value": float(series[latest])})

    out = pd.DataFrame(rows, columns=TIDY_COLS)
    if not out.empty:
        save_cache(out, "weo")
    return out


def fetch_weo(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and (c := load_cache("weo")) is not None:
        return c

    path = _find_weo_file()
    if path is None:
        print("[weo] No local flat file -- falling back to IMF DataMapper API.")
        d = _fetch_datamapper()
        if d.empty:
            _print_download_help()
        return d

    # WEO files are latin-1 / tab-separated, with thousands separators and 'n/a'.
    raw = pd.read_csv(
        path, sep="\t", encoding="latin-1", thousands=",",
        na_values=["n/a", "--", ""], low_memory=False,
    )
    raw.columns = [str(c).strip() for c in raw.columns]

    # Identify the key columns (names vary slightly across vintages).
    iso_col = next((c for c in raw.columns if c.upper() == "ISO"), None)
    subj_col = next((c for c in raw.columns if "Subject Code" in c), None)
    if iso_col is None or subj_col is None:
        raise RuntimeError(
            f"Could not find ISO / Subject Code columns in {path.name}. "
            f"Columns seen: {list(raw.columns)[:8]}..."
        )

    wanted = set(config.WEO_INDICATORS.keys())
    sub = raw[raw[subj_col].isin(wanted) & raw[iso_col].isin(config.ISO3_LIST)].copy()

    year_cols = [c for c in sub.columns if c.isdigit() and len(c) == 4]
    long = sub.melt(
        id_vars=[iso_col, subj_col], value_vars=year_cols,
        var_name="year", value_name="value",
    )
    long = long.rename(columns={iso_col: "iso3", subj_col: "indicator"})
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])

    # Pin vintage or take most-recent available (historical, not forecast) per series.
    if config.VINTAGE_YEAR is not None:
        long = long[long["year"] == config.VINTAGE_YEAR]
    else:
        long = (
            long.sort_values("year")
            .groupby(["iso3", "indicator"], as_index=False)
            .last()
        )

    name_map = {code: meta["name"] for code, meta in config.WEO_INDICATORS.items()}
    long["indicator"] = long["indicator"].map(name_map).fillna(long["indicator"])

    out = long[TIDY_COLS].reset_index(drop=True)
    save_cache(out, "weo")
    return out


if __name__ == "__main__":
    d = fetch_weo(use_cache=False)
    if d.empty:
        print("[weo] empty -- see download instructions above.")
    else:
        print(f"WEO: {len(d)} rows, {d['iso3'].nunique()} countries, "
              f"{d['indicator'].nunique()} indicators")
