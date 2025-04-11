import numpy as np
import pandas as pd
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

    def plot_explained_variance(self):
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(self.explained_variance) + 1), self.explained_variance, tick_label=[f'PC{i}' for i in range(1, len(self.explained_variance) + 1)])
        plt.title("Explained Variance by PCA Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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

    def get_feature_dataframe(self, index=None):
        """
        Returns PCA components as a DataFrame with column names PC1, PC2, ...
        """
        columns = [f"PC{i+1}" for i in range(self.components.shape[1])]
        return pd.DataFrame(self.components, index=index, columns=columns)

    def save_features(self, filepath, index=None):
        """
        Saves PCA components as CSV for ML regression.
        """
        df = self.get_feature_dataframe(index=index)
        df.to_csv(filepath)
        print(f"[✓] PCA features saved to: {filepath}")