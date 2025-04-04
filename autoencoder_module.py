import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(42)


class AutoencoderReducer:
    """
    Deep autoencoder for dimensionality reduction of returns matrix.
    Includes optional scaling and a 5-layer architecture:
    Input(25) → Dense(16) → Dense(5) → Dense(16) → Output(25)
    """

    def __init__(self, returns_matrix, encoding_dim=5, scale_factor=1.0, epochs=100, batch_size=32):
        self.original_returns = returns_matrix
        self.scale_factor = scale_factor
        self.encoding_dim = encoding_dim
        self.input_dim = returns_matrix.shape[1]
        self.scaled_returns = self.original_returns * self.scale_factor

        self._build_model()
        self._train(epochs, batch_size)

    def _build_model(self):
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        x = Dense(16, activation='tanh')(input_layer)
        encoded = Dense(self.encoding_dim, activation='tanh')(x)

        # Decoder
        x = Dense(16, activation='tanh')(encoded)
        decoded = Dense(self.input_dim, activation='linear')(x)

        # Full Autoencoder
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.encoder = Model(inputs=input_layer, outputs=encoded)

        # Standalone Decoder
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer1 = self.autoencoder.layers[-2]
        decoder_layer2 = self.autoencoder.layers[-1]
        x = decoder_layer1(encoded_input)
        output = decoder_layer2(x)
        self.decoder = Model(inputs=encoded_input, outputs=output)

        # Compile
        self.autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def _train(self, epochs, batch_size):
        self.history = self.autoencoder.fit(
            self.scaled_returns, self.scaled_returns,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0
        )

    def get_latent_features(self):
        return self.encoder.predict(self.scaled_returns)

    def inverse_transform(self, latent_matrix):
        reconstructed_scaled = self.decoder.predict(latent_matrix)
        return reconstructed_scaled / self.scale_factor

    def plot_reconstruction_loss(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.title("Reconstruction Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
  
