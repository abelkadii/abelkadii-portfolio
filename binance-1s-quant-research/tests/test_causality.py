import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quant_research import causal_features, turning_point_labels


def synthetic_frame(n: int = 600) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    returns = 0.00005 * np.sin(t / 13.0) + 0.00002 * np.cos(t / 5.0)
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = 0.0002 + 0.00005 * (1.0 + np.sin(t / 17.0))
    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC"),
        "open": close * (1.0 - 0.00001),
        "high": close * (1.0 + spread),
        "low": close * (1.0 - spread),
        "close": close,
        "volume": 10.0 + 2.0 * (1.0 + np.sin(t / 9.0)),
    })


def test_causal_features_do_not_change_when_future_is_modified() -> None:
    frame = synthetic_frame()
    cutoff = 350
    original = causal_features(frame)
    modified_input = frame.copy()
    future = modified_input.index > cutoff
    modified_input.loc[future, ["open", "high", "low", "close"]] *= 1.25
    modified_input.loc[future, "volume"] *= 5.0
    modified = causal_features(modified_input)

    feature_columns = [
        "ret_1s", "rv_30", "rv_120", "return_z", "trend_5", "trend_30",
        "trend_30_z", "range_position", "volume_z", "peak_score", "valley_score",
    ]
    assert_frame_equal(
        original.loc[:cutoff, feature_columns],
        modified.loc[:cutoff, feature_columns],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_turning_point_labels_are_boolean_and_future_bounded() -> None:
    labeled = turning_point_labels(causal_features(synthetic_frame()), horizon=30, move_bps=1.0)
    assert labeled["peak_label"].dtype == bool
    assert labeled["valley_label"].dtype == bool
    assert labeled["future_return_30"].tail(30).isna().all()
