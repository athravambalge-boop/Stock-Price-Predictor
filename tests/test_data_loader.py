import pandas as pd

import data_loader


def test_load_data_flattens_single_ticker_multiindex(monkeypatch):
    cols = pd.MultiIndex.from_product([
        ["Open", "High", "Low", "Close", "Volume"],
        ["TCS.NS"],
    ])
    frame = pd.DataFrame([[1, 2, 0.5, 1.5, 1000]], columns=cols)

    def fake_download(*args, **kwargs):
        return frame

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    out = data_loader.load_data("TCS.NS", start="2024-01-01", end="2024-02-01")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_data_retries_on_failure(monkeypatch):
    calls = {"count": 0}

    def flaky_download(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary failure")
        return pd.DataFrame(
            {
                "Open": [1.0],
                "High": [2.0],
                "Low": [0.5],
                "Close": [1.5],
                "Volume": [1000],
            }
        )

    monkeypatch.setattr(data_loader.yf, "download", flaky_download)

    out = data_loader.load_data("TCS.NS", retries=3, backoff_seconds=0.0)
    assert not out.empty
    assert calls["count"] == 2
