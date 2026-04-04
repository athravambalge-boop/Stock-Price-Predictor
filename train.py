import numpy as np

def train_lstm(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

def predict_lstm(model, X_test, scaler):
    pred = model.predict(X_test)
    return scaler.inverse_transform(pred)