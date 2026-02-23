from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def main():
    # Input layer (8 neurons)
    inputs = Input((8,), name='input_layer')

    # Hidden layers
    h1 = Dense(4, activation='relu', name='hidden_layer_1')(inputs)
    h2 = Dense(8, activation='relu', name='hidden_layer_2')(h1)
    h3 = Dense(4, activation='relu', name='hidden_layer_3')(h2)

    # Output layer (10 neurons)
    outputs = Dense(10, activation='softmax', name='output_layer')(h3)

    # Build model
    model = Model(inputs, outputs)

    # Show summary
    model.summary(show_trainable=True)


if __name__ == '__main__':
    main()
