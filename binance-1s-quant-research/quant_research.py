from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class ResearchConfig:
    horizon: int = 30
    move_bps: float = 8.0
    train_fraction: float = 0.70
    score_quantile: float = 0.995


def _future_rolling(series: pd.Series, window: int, op: str) -> pd.Series:
    shifted = series.shift(-1)
    rolling = shifted.rolling(window, min_periods=window)
    aggregated = rolling.min() if op == "min" else rolling.max()
    return aggregated.shift(-(window - 1))


def causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = frame.copy().sort_values("open_time").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    log_close = np.log(out["close"].clip(lower=EPS))
    ret = log_close.diff()
    out["ret_1s"] = ret
    out["rv_30"] = ret.rolling(30, min_periods=20).std()
    out["rv_120"] = ret.rolling(120, min_periods=60).std()
    out["return_z"] = ret / (out["rv_30"].shift(1) + EPS)

    out["trend_5"] = log_close.shift(1) - log_close.shift(6)
    out["trend_30"] = log_close.shift(1) - log_close.shift(31)
    out["trend_30_z"] = out["trend_30"] / (out["rv_120"].shift(1) * sqrt(30) + EPS)

    trailing_high = out["high"].rolling(60, min_periods=30).max()
    trailing_low = out["low"].rolling(60, min_periods=30).min()
    range_width = (trailing_high - trailing_low).clip(lower=EPS)
    out["range_position"] = (2.0 * (out["close"] - trailing_low) / range_width - 1.0).clip(-1.5, 1.5)

    log_volume = np.log1p(out["volume"].clip(lower=0))
    volume_mean = log_volume.rolling(120, min_periods=60).mean().shift(1)
    volume_std = log_volume.rolling(120, min_periods=60).std().shift(1)
    out["volume_z"] = (log_volume - volume_mean) / (volume_std + EPS)

    trend_strength = np.tanh(out["trend_30_z"].fillna(0.0) / 2.0)
    reversal_down = np.clip(-out["return_z"].fillna(0.0), 0.0, 4.0) / 4.0
    reversal_up = np.clip(out["return_z"].fillna(0.0), 0.0, 4.0) / 4.0
    upper_range = np.clip(out["range_position"].fillna(0.0), 0.0, 1.0)
    lower_range = np.clip(-out["range_position"].fillna(0.0), 0.0, 1.0)
    positive_trend = np.clip(trend_strength, 0.0, 1.0)
    negative_trend = np.clip(-trend_strength, 0.0, 1.0)
    volume_surprise = np.clip(out["volume_z"].fillna(0.0), 0.0, 4.0) / 4.0

    out["peak_score"] = 0.40 * upper_range + 0.30 * positive_trend + 0.20 * reversal_down + 0.10 * volume_surprise
    out["valley_score"] = 0.40 * lower_range + 0.30 * negative_trend + 0.20 * reversal_up + 0.10 * volume_surprise
    return out


def turning_point_labels(frame: pd.DataFrame, horizon: int, move_bps: float) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    out = frame.copy()
    move = move_bps / 10_000.0
    future_low = _future_rolling(out["low"], horizon, "min")
    future_high = _future_rolling(out["high"], horizon, "max")
    past_high = out["high"].rolling(horizon + 1, min_periods=horizon + 1).max()
    past_low = out["low"].rolling(horizon + 1, min_periods=horizon + 1).min()

    out[f"future_return_{horizon}"] = out["close"].shift(-horizon) / out["close"] - 1.0
    out["future_min_return"] = future_low / out["close"] - 1.0
    out["future_max_return"] = future_high / out["close"] - 1.0

    local_peak = (out["high"] >= past_high) & (out["high"] >= future_high)
    local_valley = (out["low"] <= past_low) & (out["low"] <= future_low)
    out["peak_label"] = local_peak & (out["future_min_return"] <= -move)
    out["valley_label"] = local_valley & (out["future_max_return"] >= move)
    return out


def _direction_metrics(test: pd.DataFrame, *, direction: str, threshold: float, horizon: int) -> dict[str, float | int]:
    signal = test[f"{direction}_score"] >= threshold
    label = test[f"{direction}_label"].fillna(False).astype(bool)
    signal_count = int(signal.sum())
    label_count = int(label.sum())
    true_positive = int((signal & label).sum())
    precision = true_positive / signal_count if signal_count else 0.0
    recall = true_positive / label_count if label_count else 0.0
    label_rate = float(label.mean()) if len(label) else 0.0
    signal_rate = float(signal.mean()) if len(signal) else 0.0
    lift = precision / label_rate if label_rate > 0 else 0.0

    fwd = test.loc[signal, f"future_return_{horizon}"].dropna()
    if direction == "peak":
        signed_forward = -fwd
        favorable = -test.loc[signal, "future_min_return"].dropna()
        adverse = test.loc[signal, "future_max_return"].dropna()
    else:
        signed_forward = fwd
        favorable = test.loc[signal, "future_max_return"].dropna()
        adverse = -test.loc[signal, "future_min_return"].dropna()

    return {
        "threshold": float(threshold),
        "signal_count": signal_count,
        "signal_rate": signal_rate,
        "label_count": label_count,
        "label_rate": label_rate,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "lift_vs_base_rate": lift,
        "mean_signed_forward_return": float(signed_forward.mean()) if len(signed_forward) else 0.0,
        "median_signed_forward_return": float(signed_forward.median()) if len(signed_forward) else 0.0,
        "mean_favorable_excursion": float(favorable.mean()) if len(favorable) else 0.0,
        "mean_adverse_excursion": float(adverse.mean()) if len(adverse) else 0.0,
    }


def run_baseline(frame: pd.DataFrame, config: ResearchConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    enriched = turning_point_labels(causal_features(frame), config.horizon, config.move_bps)
    usable = enriched.dropna(subset=["peak_score", "valley_score", f"future_return_{config.horizon}"]).reset_index(drop=True)
    if len(usable) < 500:
        raise ValueError("not enough usable rows; provide a longer sample")

    split = int(len(usable) * config.train_fraction)
    if split <= 0 or split >= len(usable):
        raise ValueError("train_fraction must leave non-empty train and test segments")

    train = usable.iloc[:split]
    test = usable.iloc[split:].copy()
    peak_threshold = float(train["peak_score"].quantile(config.score_quantile))
    valley_threshold = float(train["valley_score"].quantile(config.score_quantile))
    test["peak_signal"] = test["peak_score"] >= peak_threshold
    test["valley_signal"] = test["valley_score"] >= valley_threshold

    metrics: dict[str, object] = {
        "config": {"horizon": config.horizon, "move_bps": config.move_bps,
                   "train_fraction": config.train_fraction, "score_quantile": config.score_quantile},
        "rows": {"total_usable": int(len(usable)), "train": int(len(train)), "test": int(len(test))},
        "peak": _direction_metrics(test, direction="peak", threshold=peak_threshold, horizon=config.horizon),
        "valley": _direction_metrics(test, direction="valley", threshold=valley_threshold, horizon=config.horizon),
    }
    return test, metrics
