import numpy as np

from train import returns_to_price, evaluate_forecast, directional_accuracy


def test_returns_to_price_matches_expected_math():
    prev_close = np.array([[100.0], [120.0], [80.0]])
    pred_returns = np.array([[0.0], [np.log(1.10)], [np.log(0.90)]])

    out = returns_to_price(pred_returns, prev_close)
    expected = np.array([[100.0], [132.0], [72.0]])

    assert np.allclose(out, expected)


def test_evaluate_forecast_outputs_expected_keys_and_finite_values():
    y_true = np.array([[100.0], [102.0], [101.0], [105.0]])
    y_pred = np.array([[99.0], [101.0], [102.0], [104.0]])
    prev = np.array([[99.5], [100.5], [100.8], [103.2]])

    metrics = evaluate_forecast(y_true, y_pred, prev)

    assert set(['rmse', 'mae', 'mape', 'directional_accuracy', 'residuals', 'ci95']).issubset(metrics.keys())
    assert np.isfinite(metrics['rmse'])
    assert np.isfinite(metrics['mae'])
    assert np.isfinite(metrics['mape'])
    assert np.isfinite(metrics['directional_accuracy'])


def test_directional_accuracy_simple_case():
    y_true = np.array([[102.0], [98.0]])
    y_pred = np.array([[101.0], [99.0]])
    prev = np.array([[100.0], [100.0]])

    score = directional_accuracy(y_true, y_pred, prev)
    assert score == 100.0
