import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam


class VAEModel(Model):
    def __init__(self, input_dim, latent_dim):
        super(VAEModel, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Encoder
        self.encoder_dense = Dense(16, activation='tanh')
        self.z_mean_dense = Dense(latent_dim)
        self.z_log_var_dense = Dense(latent_dim)

        # Decoder
        self.decoder_dense1 = Dense(16, activation='tanh')
        self.decoder_output = Dense(input_dim)

    def encode(self, x):
        h = self.encoder_dense(x)
        z_mean = self.z_mean_dense(h)
        z_log_var = self.z_log_var_dense(h)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        x = self.decoder_dense1(z)
        return self.decoder_output(x)

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)

        # Save for later access
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z_sample = z

        # Compute VAE loss
        reconstruction_loss = tf.reduce_sum(tf.square(inputs - reconstructed), axis=1)
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        self.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
        return reconstructed


class VariationalAutoencoderReducer:
    def __init__(self, returns_matrix, latent_dim=5, epochs=100, batch_size=32):
        self.original_returns = returns_matrix
        self.latent_dim = latent_dim
        self.input_dim = returns_matrix.shape[1]

        self.scaler = StandardScaler()
        self.scaled_returns = self.scaler.fit_transform(self.original_returns)

        self.vae_model = VAEModel(self.input_dim, self.latent_dim)
        self.vae_model.compile(optimizer=Adam(learning_rate=0.001))
        self.history = self.vae_model.fit(
            self.scaled_returns, self.scaled_returns,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0
        )

        # Extract encoder and decoder components
        self.encoder = lambda x: self.vae_model.encode(x)[0].numpy()
        self.encoder_full = lambda x: self.vae_model.reparameterize(*self.vae_model.encode(x)).numpy()
        self.decoder = lambda z: self.vae_model.decode(
            tf.convert_to_tensor(z.reshape(1, -1)) if z.ndim == 1 else tf.convert_to_tensor(z)
        ).numpy()


    def get_z_mean(self):
        return self.encoder(self.scaled_returns)

    def get_sampled_latent(self):
        return self.encoder_full(self.scaled_returns)

    def inverse_transform(self, latent_matrix):
        reconstructed_scaled = self.decoder(latent_matrix)

        # Ensure input is 2D: (n_samples, n_features)
        if reconstructed_scaled.ndim == 1:
            reconstructed_scaled = reconstructed_scaled.reshape(1, -1)
        return self.scaler.inverse_transform(reconstructed_scaled)

    def plot_reconstruction_loss(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.title("VAE Reconstruction Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_feature_dataframe(self, use_sampled=True, index=None):
        latent = self.get_sampled_latent() if use_sampled else self.get_z_mean()
        columns = [f"Z{i+1}" for i in range(latent.shape[1])]
        return pd.DataFrame(latent, index=index, columns=columns)

    def save_features(self, filepath, use_sampled=True, index=None):
        df = self.get_feature_dataframe(use_sampled=use_sampled, index=index)
        df.to_csv(filepath)
        print(f"[✓] VAE latent features saved to: {filepath}")

    def sample_latent_vectors(self, n_samples=1):
        return np.random.normal(size=(n_samples, self.latent_dim))
    
    def sample_from_aggregated_posterior(self, n_samples=1000):
        """
        Samples from the aggregated posterior distribution over the training window:
        z ~ N(mean(z_mean), mean(exp(z_log_var)))
        Returns array of shape (n_samples, latent_dim)
        """
        z_mean, z_log_var = self.vae_model.encode(self.scaled_returns)

        # Collapse to single mean and variance across all T samples
        mu = tf.reduce_mean(z_mean, axis=0).numpy()                     # shape: (latent_dim,)
        sigma = tf.reduce_mean(tf.exp(z_log_var), axis=0).numpy() ** 0.5  # stddev: shape (latent_dim,)

        # Sample from the aggregated distribution
        epsilon = np.random.normal(size=(n_samples, self.latent_dim))   # shape: (n_samples, latent_dim)
        z_samples = mu + epsilon * sigma

        return z_samples


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Lambda
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import backend as K
# from tensorflow.keras import Layer
# from tensorflow.keras.losses import mse

# class SquareLayer(Layer):
#     def call(self, x):
#         return tf.square(x)
    
# class ExpLayer(Layer):
#     def call(self, x):
#         return tf.exp(x)

# class ReduceMeanLayer(Layer):
#     def call(self, x):
#         return tf.reduce_mean(x)

# class ReduceSumLayer(Layer):
#     def call(self, x, axis):
#         return tf.reduce_sum(x, axis=axis)

# class VariationalAutoencoderReducer:
#     def __init__(self, returns_matrix, latent_dim=5, epochs=100, batch_size=32):
#         self.original_returns = returns_matrix
#         self.latent_dim = latent_dim
#         self.input_dim = returns_matrix.shape[1]

#         self.scaler = StandardScaler()
#         self.scaled_returns = self.scaler.fit_transform(self.original_returns)

#         self._build_model()
#         self._train(epochs, batch_size)

#     def _sampling(self, args):
#         z_mean, z_log_var = args
#         epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon

#     def _build_model(self):
#         # === Encoder === #
#         inputs = Input(shape=(self.input_dim,))
#         h = Dense(16, activation='tanh')(inputs)
#         z_mean = Dense(self.latent_dim, name="z_mean")(h)
#         z_log_var = Dense(self.latent_dim, name="z_log_var")(h)

#         def sampling(args):
#             z_mean, z_log_var = args
#             epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
#             return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#         z = Lambda(sampling, name="z")([z_mean, z_log_var])

#         # === Decoder === #
#         latent_inputs = Input(shape=(self.latent_dim,))
#         x = Dense(16, activation='tanh')(latent_inputs)
#         outputs = Dense(self.input_dim, activation='linear')(x)

#         decoder = Model(latent_inputs, outputs, name="decoder")
#         outputs_decoded = decoder(z)

#         # === VAE model === #
#         vae = Model(inputs, outputs_decoded, name="vae")

#         # === Correct symbolic loss === #
#         reconstruction_loss = ReduceSumLayer()(
#             SquareLayer()(inputs - outputs_decoded), axis=1
#         )  # shape (batch_size,)
#         kl_loss = -0.5 * ReduceSumLayer()(
#             1 + z_log_var - SquareLayer()(z_mean) - ExpLayer()(z_log_var), axis=1
#         )  # shape (batch_size,)
#         total_vae_loss = ReduceMeanLayer()(reconstruction_loss + kl_loss)

#         vae.add_loss(total_vae_loss)
#         vae.compile(optimizer=Adam(learning_rate=0.001))

#         # Save models
#         self.encoder = Model(inputs, z_mean)
#         self.encoder_full = Model(inputs, z)
#         self.decoder = decoder
#         self.vae = vae

#     # def _build_model(self):
#     #     # === Encoder === #
#     #     inputs = Input(shape=(self.input_dim,))
#     #     h = Dense(16, activation='tanh')(inputs)
#     #     z_mean = Dense(self.latent_dim, name="z_mean")(h)
#     #     z_log_var = Dense(self.latent_dim, name="z_log_var")(h)
#     #     z = Lambda(self._sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])

#     #     # Save encoder outputs for reuse
#     #     self._inputs = inputs
#     #     self._z_mean = z_mean
#     #     self._z_log_var = z_log_var
#     #     self._z = z

#     #     # === Decoder === #
#     #     latent_inputs = Input(shape=(self.latent_dim,))
#     #     x = Dense(16, activation='tanh')(latent_inputs)
#     #     outputs = Dense(self.input_dim, activation='linear')(x)

#     #     # Models
#     #     self.encoder = Model(inputs, z_mean)  # z_mean used for deterministic mapping
#     #     self.encoder_full = Model(inputs, z)  # z used for full sampled encoder
#     #     self.decoder = Model(latent_inputs, outputs)

#     #     # VAE model
#     #     vae_outputs = self.decoder(z)
#     #     self.vae = Model(inputs, vae_outputs)

#     #     # === VAE Loss === #
#     #     reconstruction_loss = tf.keras.losses.mse(inputs, vae_outputs)
#     #     reconstruction_loss *= self.input_dim
#     #     kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#     #     vae_loss = K.mean(reconstruction_loss + kl_loss)

#     #     self.vae.add_loss(vae_loss)
#     #     self.vae.compile(optimizer=Adam(learning_rate=0.001))

#     def _train(self, epochs, batch_size):
#         self.history = self.vae.fit(
#             self.scaled_returns, None,
#             epochs=epochs,
#             batch_size=batch_size,
#             shuffle=True,
#             verbose=0
#         )

#     def get_z_mean(self):
#         return self.encoder.predict(self.scaled_returns)

#     def get_sampled_latent(self):
#         return self.encoder_full.predict(self.scaled_returns)

#     def inverse_transform(self, latent_matrix):
#         reconstructed_scaled = self.decoder.predict(latent_matrix)
#         return self.scaler.inverse_transform(reconstructed_scaled)

#     def plot_reconstruction_loss(self, save_path=None):
#         plt.figure(figsize=(8, 4))
#         plt.plot(self.history.history['loss'], label='Training Loss')
#         plt.title("VAE Reconstruction Loss over Epochs")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path)
#             plt.close()
#         else:
#             plt.show()

#     def get_feature_dataframe(self, use_sampled=True, index=None):
#         latent = self.get_sampled_latent() if use_sampled else self.get_z_mean()
#         columns = [f"Z{i+1}" for i in range(latent.shape[1])]
#         return pd.DataFrame(latent, index=index, columns=columns)

#     def save_features(self, filepath, use_sampled=True, index=None):
#         df = self.get_feature_dataframe(use_sampled=use_sampled, index=index)
#         df.to_csv(filepath)
#         print(f"[✓] VAE latent features saved to: {filepath}")

#     def sample_latent_vectors(self, n_samples=1):
#         """Draw latent vectors from N(0, I) for Monte Carlo-style testing."""
#         return np.random.normal(size=(n_samples, self.latent_dim))


