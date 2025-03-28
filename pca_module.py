import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAReducer:
    """
    Class to perform Principal Component Analysis (PCA) on a return matrix.
    """

    def __init__(self, returns_matrix, n_components=0.95):
        self.returns_matrix = returns_matrix
        self.pca = PCA(n_components=n_components)
        self.components = self.pca.fit_transform(returns_matrix)
        self.explained_variance = self.pca.explained_variance_ratio_

    def get_components(self):
        return self.components

    def inverse_transform(self, perturbed_components):
        return self.pca.inverse_transform(perturbed_components)

    def get_explained_variance(self):
        return self.explained_variance

    def plot_cumulative_variance(self):
        cumulative_variance = np.cumsum(self.explained_variance)
        plt.figure(figsize=(8, 4))
        plt.plot(cumulative_variance, marker='o')
        plt.title("Cumulative Explained Variance by Principal Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()