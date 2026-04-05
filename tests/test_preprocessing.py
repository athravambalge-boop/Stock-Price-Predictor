import numpy as np
import pandas as pd

from preprocessing import add_features, build_sequence_dataset


def _sample_ohlcv(n=180):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = np.linspace(100, 140, n) + np.sin(np.arange(n) / 7.0)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1_000_000, 1_500_000, n),
        },
        index=idx,
    )


def test_add_features_no_nan_after_dropna():
    data = _sample_ohlcv()
    enriched = add_features(data)

    assert not enriched.empty
    assert enriched.isna().sum().sum() == 0


def test_sequence_alignment_dates_and_targets():
    data = add_features(_sample_ohlcv())
    feature_cols = [
        'Close', 'MA20', 'MA50', 'MA100', 'EMA20', 'RSI',
        'ATR14', 'BollingerUpper', 'BollingerLower', 'BollingerWidth',
        'VolumeChange', 'OBV', 'Volatility20'
    ]

    bundle = build_sequence_dataset(data, feature_cols=feature_cols, target_col='LogReturn', sequence_length=30)

    assert len(bundle['X_raw']) == len(bundle['y_raw'])
    assert len(bundle['dates']) == len(bundle['y_raw'])

    first_target_idx = data.index[30]
    assert bundle['dates'][0] == first_target_idx
    assert np.isclose(bundle['y_price'][0, 0], data.loc[first_target_idx, 'Close'])
    prev_idx = data.index[29]
    assert np.isclose(bundle['prev_close'][0, 0], data.loc[prev_idx, 'Close'])
