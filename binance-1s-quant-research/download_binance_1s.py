from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume",
    "taker_buy_base_volume", "taker_buy_quote_volume",
]


def utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def download_klines(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
                    *, pause_seconds: float = 0.05, timeout_seconds: float = 20.0) -> pd.DataFrame:
    if end <= start:
        raise ValueError("end must be later than start")

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[object]] = []

    with requests.Session() as session:
        while start_ms < end_ms:
            response = session.get(
                BINANCE_KLINES_URL,
                params={"symbol": symbol.upper(), "interval": interval, "startTime": start_ms,
                        "endTime": end_ms - 1, "limit": 1000},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break

            rows.extend(row for row in batch if int(row[0]) < end_ms)
            next_start = int(batch[-1][6]) + 1
            if next_start <= start_ms:
                raise RuntimeError("Binance pagination did not advance")
            start_ms = next_start
            if pause_seconds > 0:
                time.sleep(pause_seconds)

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        return frame

    frame = frame.drop(columns=["ignore"])
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric)
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], downcast="integer")
    return frame.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def save_frame(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    else:
        raise ValueError("output must end in .parquet or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance OHLCV klines")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1s")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pause-seconds", type=float, default=0.05)
    args = parser.parse_args()

    frame = download_klines(
        args.symbol, args.interval, utc_timestamp(args.start), utc_timestamp(args.end),
        pause_seconds=args.pause_seconds,
    )
    save_frame(frame, args.output)
    print(f"Saved {len(frame):,} candles to {args.output}")


if __name__ == "__main__":
    main()
