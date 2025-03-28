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
