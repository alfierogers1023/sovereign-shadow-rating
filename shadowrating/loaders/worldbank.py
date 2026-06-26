"""
World Bank loader (WDI + WGI) via the wbgapi package.

WDI lives in database 2, WGI in database 3. We pull the most-recent non-empty
value per country per series (or a pinned vintage if config.VINTAGE_YEAR is set).

Network: hits api.worldbank.org. If that's unreachable you'll get a clear
RuntimeError rather than a silent empty frame.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from . import TIDY_COLS, save_cache, load_cache

WDI_DB = 2
WGI_DB = 3


def _fetch(indicators: dict[str, dict], db: int) -> pd.DataFrame:
    try:
        import wbgapi as wb
    except ImportError as e:
        raise RuntimeError("wbgapi not installed. Run: pip install wbgapi") from e

    codes = list(indicators.keys())
    economies = config.ISO3_LIST

    # mrnev=1 -> most recent non-empty value per series/economy.
    # If a fixed vintage is requested, ask for that single year instead.
    kwargs = dict(db=db, skipBlanks=True)
    if config.VINTAGE_YEAR is None:
        kwargs["mrnev"] = 1
    else:
        kwargs["time"] = [config.VINTAGE_YEAR]

    # Deliberately wb.data.fetch(), not wb.data.DataFrame(): DataFrame()
    # silently drops the vintage year when mrnev is used (confirmed by
    # inspecting both), which would make every WDI/WGI cell's actual age
    # unknowable -- exactly the data-quality question Phase 4's divergence
    # table needs to answer. fetch() returns it as a 'YR<year>' string on
    # every record regardless of mrnev/pinned-vintage mode.
    try:
        rows = list(wb.data.fetch(codes, economies, **kwargs))
    except Exception as e:  # noqa: BLE001 -- surface any network/API failure clearly
        raise RuntimeError(
            f"World Bank API call failed (db={db}). Check connectivity to "
            f"api.worldbank.org. Underlying error: {e}"
        ) from e

    if not rows:
        return pd.DataFrame(columns=TIDY_COLS)

    long = pd.DataFrame(rows)
    long = long.rename(columns={"economy": "iso3", "series": "indicator"})
    long["year"] = (
        long["time"].astype(str).str.extract(r"(\d{4})").astype("float").astype("Int64")
    )

    # Friendly indicator names.
    name_map = {code: meta["name"] for code, meta in indicators.items()}
    long["indicator"] = long["indicator"].map(name_map).fillna(long["indicator"])

    long = long.dropna(subset=["value"])
    return long[TIDY_COLS].reset_index(drop=True)


def fetch_wdi(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and (c := load_cache("wdi")) is not None:
        return c
    df = _fetch(config.WDI_INDICATORS, WDI_DB)
    save_cache(df, "wdi")
    return df


def fetch_wgi(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and (c := load_cache("wgi")) is not None:
        return c
    df = _fetch(config.WGI_INDICATORS, WGI_DB)
    save_cache(df, "wgi")
    return df


if __name__ == "__main__":
    for name, fn in [("WDI", fetch_wdi), ("WGI", fetch_wgi)]:
        d = fn(use_cache=False)
        print(f"{name}: {len(d)} rows, {d['iso3'].nunique()} countries, "
              f"{d['indicator'].nunique()} indicators")
