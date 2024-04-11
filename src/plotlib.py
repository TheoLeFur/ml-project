import matplotlib.pyplot as plt
import numpy as np


class PlotLib:

    @staticmethod
    def plot_feature_values(data: np.ndarray) -> None:

        """
        Scatter plot of the feature's values.
        Args:
            data: Array of shape (N, D)

        Returns: None

        """

        N = data.shape[0]
        n_features = data.shape[1]

        # Feature values plot
        fig, axs = plt.subplots(5, 1, figsize=(8, 12))

        for i in range(n_features):
            axs[i].scatter(range(N), data[:, i])
            axs[i].set_title(f'Feature {i + 1}')
            axs[i].set_xlabel('Image Index')
            axs[i].set_ylabel(f'Feature {i + 1} value')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_distribution(data: np.ndarray) -> None:
        """
        Distribution of the values of each feature.

        Args:
            data: Array of shape (N, D)

        Returns: None

        """

        n_features = data.shape[1]

        # Feature distribution histogram
        fig, axs = plt.subplots(n_features, 1, figsize=(8, 12))

        for i in range(n_features):
            axs[i].hist(data[:, i], bins=30)
            axs[i].set_title(f'Feature {i + 1} Distribution')
            axs[i].set_xlabel(f'Feature {i + 1} value')
            axs[i].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()
