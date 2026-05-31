from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.optimizers import Adam

def build_lstm(input_shape, seed=42):
    """Enhanced LSTM architecture with bidirectional layers and increased capacity"""
    model = Sequential()
    kernel_init = GlorotUniform(seed=seed)
    recurrent_init = Orthogonal(seed=seed)

    # First bidirectional LSTM layer
    model.add(
        Bidirectional(
            LSTM(
                128,
                return_sequences=True,
                input_shape=input_shape,
                kernel_initializer=kernel_init,
                recurrent_initializer=recurrent_init,
            )
        )
    )
    model.add(Dropout(0.3))

    # Second bidirectional LSTM layer
    model.add(
        Bidirectional(
            LSTM(
                64,
                return_sequences=True,
                kernel_initializer=kernel_init,
                recurrent_initializer=recurrent_init,
            )
        )
    )
    model.add(Dropout(0.3))

    # Third LSTM layer (unidirectional)
    model.add(
        LSTM(
            32,
            kernel_initializer=kernel_init,
            recurrent_initializer=recurrent_init,
        )
    )
    model.add(Dropout(0.2))

    # Dense layers with more capacity
    model.add(Dense(64, activation='relu', kernel_initializer=kernel_init))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_initializer=kernel_init))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer=kernel_init))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def build_lstm_attention(input_shape, seed=42):
    """LSTM with attention mechanism for better feature focus"""
    kernel_init = GlorotUniform(seed=seed)
    recurrent_init = Orthogonal(seed=seed)
    
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM
    x = Bidirectional(
        LSTM(96, return_sequences=True, kernel_initializer=kernel_init, recurrent_initializer=recurrent_init)
    )(inputs)
    x = Dropout(0.3)(x)
    
    # Second Bidirectional LSTM
    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_initializer=kernel_init, recurrent_initializer=recurrent_init)
    )(x)
    x = Dropout(0.3)(x)
    
    # Self-attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Final LSTM
    x = LSTM(32, kernel_initializer=kernel_init, recurrent_initializer=recurrent_init)(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(64, activation='relu', kernel_initializer=kernel_init)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', kernel_initializer=kernel_init)(x)
    outputs = Dense(1, kernel_initializer=kernel_init)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=0.0008)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model