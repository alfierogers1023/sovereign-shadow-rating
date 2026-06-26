"""
FRED long-term government bond yields (validation / early-warning signal).

Series pattern IRLTLT01{CC}M156N -- monthly 10y benchmark yields, OECD coverage.
We pull the latest available monthly yield per country and compute the spread
vs the benchmark (Germany by default). EM names are mostly absent from FRED;
those gaps are expected and get filled later from a dedicated spread feed.

Requires a free FRED API key. Get one at
https://fredaccount.stlouisfed.org/apikeys and expose it as:
    export FRED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
"""
from __future__ import annotations

import os

import pandas as pd

from .. import config
from . import save_cache, load_cache

YIELD_COLS = ["iso3", "yield_pct", "yield_date", "spread_vs_benchmark_bps"]


def _get_client():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return None
    try:
        from fredapi import Fred
    except ImportError as e:
        raise RuntimeError("fredapi not installed. Run: pip install fredapi") from e
    return Fred(api_key=key)


def fetch_yields(use_cache: bool = True) -> pd.DataFrame:
    """Latest monthly yield + spread vs benchmark for the FRED-covered subset."""
    if use_cache and (c := load_cache("fred_yields")) is not None:
        return c

    fred = _get_client()
    if fred is None:
        print(
            "[fred] FRED_API_KEY not set -- skipping yields. "
            "Get a free key at https://fredaccount.stlouisfed.org/apikeys "
            "and `export FRED_API_KEY=...` to enable the market-spread cross-check."
        )
        return pd.DataFrame(columns=YIELD_COLS)

    rows = []
    for iso3, cc in config.FRED_COUNTRY_CODES.items():
        series_id = config.FRED_YIELD_PATTERN.format(cc=cc)
        try:
            s = fred.get_series(series_id).dropna()
        except Exception as e:  # noqa: BLE001 -- one bad series shouldn't kill the run
            print(f"[fred] {iso3} ({series_id}) failed: {e}")
            continue
        if s.empty:
            continue
        rows.append({"iso3": iso3, "yield_pct": float(s.iloc[-1]),
                     "yield_date": s.index[-1].date().isoformat()})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=YIELD_COLS)

    bench = df.loc[df["iso3"] == config.FRED_BENCHMARK, "yield_pct"]
    bench_val = float(bench.iloc[0]) if not bench.empty else None
    df["spread_vs_benchmark_bps"] = (
        (df["yield_pct"] - bench_val) * 100 if bench_val is not None else pd.NA
    )

    df = df[YIELD_COLS].reset_index(drop=True)
    save_cache(df, "fred_yields")
    return df


if __name__ == "__main__":
    d = fetch_yields(use_cache=False)
    print(d.to_string(index=False) if not d.empty else "[fred] no data (key/network).")
