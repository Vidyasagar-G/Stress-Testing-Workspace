import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class AutoencoderReducer:
    """
    Class to perform dimensionality reduction using Autoencoder on a return matrix.
    """

    def __init__(self, returns_matrix, encoding_dim=5, epochs=100, batch_size=32):
        self.returns_matrix = returns_matrix
        self.encoding_dim = encoding_dim
        self.input_dim = returns_matrix.shape[1]
        self._build_model()
        self._train(epochs, batch_size)

    def _build_model(self):
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)

        # Decoder
        decoded = Dense(self.input_dim, activation='linear')(encoded)

        # Autoencoder
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.encoder = Model(inputs=input_layer, outputs=encoded)

        # Decoder model for reconstruction from latent space
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

        self.autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def _train(self, epochs, batch_size):
        self.history = self.autoencoder.fit(
            self.returns_matrix, self.returns_matrix,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0  # Set to 1 if you want training output
        )

    def get_latent_features(self):
        return self.encoder.predict(self.returns_matrix)

    def inverse_transform(self, latent_matrix):
        return self.decoder.predict(latent_matrix)

    def plot_reconstruction_loss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.title("Reconstruction Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
