from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_research import ResearchConfig, run_baseline


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("input must be .parquet or .csv")
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    return frame.sort_values("open_time").reset_index(drop=True)


def save_diagnostic(test: pd.DataFrame, output: Path, max_points: int = 20_000) -> None:
    plot_frame = test.iloc[-max_points:].copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_frame["open_time"], plot_frame["close"], linewidth=0.9, label="close")
    peaks = plot_frame[plot_frame["peak_signal"]]
    valleys = plot_frame[plot_frame["valley_signal"]]
    ax.scatter(peaks["open_time"], peaks["close"], marker="v", s=28, label="peak signal")
    ax.scatter(valleys["open_time"], valleys["close"], marker="^", s=28, label="valley signal")
    ax.set_title("Out-of-sample turning-point candidates")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-aware turning-point research")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--move-bps", type=float, default=8.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--score-quantile", type=float, default=0.995)
    args = parser.parse_args()

    test, metrics = run_baseline(
        load_frame(args.input),
        ResearchConfig(args.horizon, args.move_bps, args.train_fraction, args.score_quantile),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test.to_parquet(args.output_dir / "signals.parquet", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_diagnostic(test, args.output_dir / "diagnostic.png")
    print(json.dumps(metrics, indent=2))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
