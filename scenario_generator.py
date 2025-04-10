import numpy as np

class PCAScenarioGenerator:
    """
    Class for generating deterministic stress scenarios by perturbing PCA components.
    """

    def __init__(self, components):
        self.original_components = components
        self.components = components.copy()
        self.n_timesteps, self.n_components = components.shape

    def apply_single_component_shift(self, pc_index=0, sigma_multiplier=2.0):
        perturbed = self.components.copy()
        std_dev = np.std(self.components[:, pc_index])
        perturbed[:, pc_index] += sigma_multiplier * std_dev
        return perturbed

    def apply_multi_component_shift(self, shift_vector):
        perturbed = self.components.copy()
        for i, multiplier in enumerate(shift_vector):
            std_dev = np.std(self.components[:, i])
            perturbed[:, i] += multiplier * std_dev
        return perturbed


class MonteCarloPCAGenerator:
    """
    Class for generating Monte Carlo stress scenarios using multivariate Gaussian sampling.
    """

    def __init__(self, components):
        self.mean = np.mean(components, axis=0)
        self.cov = np.cov(components, rowvar=False)

    def generate_scenarios(self, n_simulations=1000):
        return np.random.multivariate_normal(self.mean, self.cov, n_simulations)

class AEScenarioGenerator:
    """
    Class for generating deterministic stress scenarios by perturbing Autoencoder latent features.
    """

    def __init__(self, latent_features):
        self.original_latent = latent_features
        self.latent = latent_features.copy()
        self.n_timesteps, self.n_features = latent_features.shape

    def apply_single_latent_shift(self, ae_index=0, sigma_multiplier=2.0):
        perturbed = self.latent.copy()
        std_dev = np.std(self.latent[:, ae_index])
        perturbed[:, ae_index] += sigma_multiplier * std_dev
        return perturbed

    def apply_multi_latent_shift(self, shift_vector):
        perturbed = self.latent.copy()
        for i, multiplier in enumerate(shift_vector):
            std_dev = np.std(self.latent[:, i])
            perturbed[:, i] += multiplier * std_dev
        return perturbed


class MonteCarloAEGenerator:
    """
    Class for generating Monte Carlo stress scenarios using multivariate Gaussian sampling on latent features.
    """

    def __init__(self, latent_features):
        self.mean = np.mean(latent_features, axis=0)
        self.cov = np.cov(latent_features, rowvar=False)

    def generate_scenarios(self, n_simulations=1000):
        return np.random.multivariate_normal(self.mean, self.cov, n_simulations)
