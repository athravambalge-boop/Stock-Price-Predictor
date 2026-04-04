import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
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


def train_lstm(model, X_train, y_train, validation_split=0.15, epochs=60, batch_size=32):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
        ModelCheckpoint('best_lstm.keras', monitor='val_loss', save_best_only=True, verbose=0),
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

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': dir_acc,
        'residuals': residuals,
    }


def baseline_predictions(X_train, y_train, X_test, prev_close_test):
    # Naive baseline: next close equals previous close
    naive_pred = np.array(prev_close_test).reshape(-1, 1)

    # Flatten sequences for classical models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    lr = LinearRegression()
    lr.fit(X_train_flat, y_train.reshape(-1))
    lr_pred = lr.predict(X_test_flat).reshape(-1, 1)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train.reshape(-1))
    rf_pred = rf.predict(X_test_flat).reshape(-1, 1)

    return {
        'naive_close': naive_pred,
        'linear_return': lr_pred,
        'rf_return': rf_pred,
    }


def walk_forward_cv_rmse(build_model_fn, X_train, y_train, target_scaler, n_splits=3):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rmse = []

    for train_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_model_fn(X_train.shape[1:])
        model.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0)

        y_val_pred_scaled = model.predict(X_val, verbose=0)
        y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled)
        y_val_true = target_scaler.inverse_transform(y_val)
        rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
        fold_rmse.append(rmse)

    return {
        'fold_rmse': fold_rmse,
        'mean_rmse': float(np.mean(fold_rmse)) if fold_rmse else np.nan,
    }