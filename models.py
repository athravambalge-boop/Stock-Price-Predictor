from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform, Orthogonal

def build_lstm(input_shape, seed=42):
    model = Sequential()
    kernel_init = GlorotUniform(seed=seed)
    recurrent_init = Orthogonal(seed=seed)

    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=input_shape,
            kernel_initializer=kernel_init,
            recurrent_initializer=recurrent_init,
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            50,
            kernel_initializer=kernel_init,
            recurrent_initializer=recurrent_init,
        )
    )
    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer=kernel_init))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model