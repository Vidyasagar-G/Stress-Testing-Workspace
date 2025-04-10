import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

class AutoencoderReducer:
    """
    Deep autoencoder for dimensionality reduction of returns matrix.
    Architecture: Input(25) → Dense(16) → Dense(5) → Dense(16) → Output(25)
    """

    def __init__(self, returns_matrix, encoding_dim=5, epochs=100, batch_size=32):
        self.original_returns = returns_matrix
        self.encoding_dim = encoding_dim
        self.input_dim = returns_matrix.shape[1]

        # === Standardize input === #
        self.scaler = StandardScaler()
        self.scaled_returns = self.scaler.fit_transform(self.original_returns)

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

        # Models
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.encoder = Model(inputs=input_layer, outputs=encoded)

        # Standalone decoder
        encoded_input = Input(shape=(self.encoding_dim,))
        x = self.autoencoder.layers[-2](encoded_input)
        output = self.autoencoder.layers[-1](x)
        self.decoder = Model(inputs=encoded_input, outputs=output)

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
        return self.scaler.inverse_transform(reconstructed_scaled)

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
