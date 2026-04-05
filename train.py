import os
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def _safe_mape(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return float(0 if np.isnan(mape) else mape)


def directional_accuracy(y_true_price, y_pred_price, prev_close):
    true_dir = np.sign(np.array(y_true_price).reshape(-1) - np.array(prev_close).reshape(-1))
    pred_dir = np.sign(np.array(y_pred_price).reshape(-1) - np.array(prev_close).reshape(-1))
    return float((true_dir == pred_dir).mean() * 100)


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_lstm(
    model,
    X_train,
    y_train,
    validation_split=0.15,
    epochs=60,
    batch_size=32,
    checkpoint_path='artifacts/best_lstm.keras',
):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def predict_lstm(model, X_test, target_scaler):
    pred_scaled = model.predict(X_test, verbose=0)
    return target_scaler.inverse_transform(pred_scaled)


def returns_to_price(pred_returns, prev_close):
    pred_returns = np.array(pred_returns).reshape(-1, 1)
    prev_close = np.array(prev_close).reshape(-1, 1)
    return prev_close * np.exp(pred_returns)


def evaluate_forecast(y_true_price, y_pred_price, prev_close):
    rmse = float(np.sqrt(mean_squared_error(y_true_price, y_pred_price)))
    mae = float(mean_absolute_error(y_true_price, y_pred_price))
    mape = _safe_mape(y_true_price, y_pred_price)
    dir_acc = directional_accuracy(y_true_price, y_pred_price, prev_close)
    residuals = np.array(y_true_price).reshape(-1) - np.array(y_pred_price).reshape(-1)

    ci95 = bootstrap_metric_intervals(y_true_price, y_pred_price, prev_close, n_bootstrap=300)

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': dir_acc,
        'residuals': residuals,
        'ci95': ci95,
    }


def baseline_predictions(X_train, y_train, X_test, prev_close_test, random_state=42):
    # Naive baseline: next close equals previous close
    naive_pred = np.array(prev_close_test).reshape(-1, 1)

    # Seasonal naive baseline: rolling 5-step average of previous closes.
    prev_close_vec = np.array(prev_close_test).reshape(-1)
    seasonal_pred = np.array([
        prev_close_vec[max(0, i - 4):i + 1].mean()
        for i in range(len(prev_close_vec))
    ]).reshape(-1, 1)

    # Flatten sequences for classical models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    lr = LinearRegression()
    lr.fit(X_train_flat, y_train.reshape(-1))
    lr_pred = lr.predict(X_test_flat).reshape(-1, 1)

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_flat, y_train.reshape(-1))
    rf_pred = rf.predict(X_test_flat).reshape(-1, 1)

    tree_preds = np.stack([tree.predict(X_test_flat) for tree in rf.estimators_], axis=0)
    rf_std = tree_preds.std(axis=0).reshape(-1, 1)

    return {
        'naive_close': naive_pred,
        'seasonal_naive_close': seasonal_pred,
        'linear_return': lr_pred,
        'rf_return': rf_pred,
        'rf_return_std': rf_std,
    }


def walk_forward_cv_rmse_leakage_safe(
    build_model_fn,
    X_raw,
    y_raw,
    n_splits=3,
    epochs=20,
    batch_size=32,
):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rmse = []

    for train_idx, val_idx in splitter.split(X_raw):
        X_tr_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_tr_raw, y_val_raw = y_raw[train_idx], y_raw[val_idx]

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        X_tr_2d = X_tr_raw.reshape(-1, X_tr_raw.shape[-1])
        feature_scaler.fit(X_tr_2d)

        X_tr = feature_scaler.transform(X_tr_2d).reshape(X_tr_raw.shape)
        X_val = feature_scaler.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape)

        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(y_tr_raw)
        y_tr = target_scaler.transform(y_tr_raw)
        y_val = target_scaler.transform(y_val_raw)

        model = build_model_fn(X_raw.shape[1:])
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

        y_val_pred_scaled = model.predict(X_val, verbose=0)
        y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_val_raw, y_val_pred)))
        fold_rmse.append(rmse)

    return {
        'fold_rmse': fold_rmse,
        'mean_rmse': float(np.mean(fold_rmse)) if fold_rmse else np.nan,
    }


def bootstrap_metric_intervals(y_true_price, y_pred_price, prev_close, n_bootstrap=300, random_state=42):
    y_true = np.array(y_true_price).reshape(-1)
    y_pred = np.array(y_pred_price).reshape(-1)
    prev = np.array(prev_close).reshape(-1)

    if len(y_true) < 2:
        return {
            'rmse': (np.nan, np.nan),
            'mae': (np.nan, np.nan),
            'mape': (np.nan, np.nan),
            'directional_accuracy': (np.nan, np.nan),
        }

    rng = np.random.default_rng(random_state)
    rmse_vals, mae_vals, mape_vals, da_vals = [], [], [], []

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        pv = prev[idx]

        rmse_vals.append(float(np.sqrt(mean_squared_error(yt, yp))))
        mae_vals.append(float(mean_absolute_error(yt, yp)))
        mape_vals.append(_safe_mape(yt, yp))
        da_vals.append(directional_accuracy(yt.reshape(-1, 1), yp.reshape(-1, 1), pv.reshape(-1, 1)))

    def q95(values):
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

    return {
        'rmse': q95(rmse_vals),
        'mae': q95(mae_vals),
        'mape': q95(mape_vals),
        'directional_accuracy': q95(da_vals),
    }