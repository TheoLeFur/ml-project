import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence
from src.methods.base_model import BaseModel
from src.methods.knn import KNN
from src.methods.logistic_regression import LogisticRegression


class PlotLib:

    @staticmethod
    def plot_feature_values(data: np.ndarray, path: Optional[str] = None) -> None:

        """
        Scatter plot of the feature's values.
        Args:
            path: Provide path and filename to save the plot, default to None
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
        if path is not None:
            plt.savefig(path)
        plt.show()

    @staticmethod
    def plot_feature_distribution(data: np.ndarray, path: Optional[str] = None) -> None:
        """
        Distribution of the values of each feature.

        Args:
            path: Provide path and filename to save the plot, default to None
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

        if path is not None:
            plt.savefig(path)

        plt.show()

    @staticmethod
    def plot_loss_against_hyperparam_val(param_grid: Sequence[BaseModel.Hyperparameters], loss_train: np.ndarray,
                                         path: Optional[str] = None) -> None:
        """
        Plots the training and testing loss against the hyperparameter values on the same figure.

        Args:
            path: Path to file where we want to save the plot, defaulted to None
            param_grid: Sequence of hyperparameter values.
            loss_train: Numpy array containing training loss values.

        Returns:
            None
        """

        if isinstance(param_grid[0], KNN.KNNHyperparameters):
            param_grid = list(map(lambda t: t.k, param_grid))
        elif isinstance(param_grid[0], LogisticRegression.LRHyperparameters):
            param_grid = list(map(lambda t: t.lr, param_grid))
        else:
            raise NotImplementedError('Unknown hyperparameter instance')

        plt.plot(param_grid, loss_train, label='Training Loss')
        plt.xlabel('Hyperparameter Value')
        plt.ylabel('Loss')
        plt.title('Loss vs. Hyperparameter Value')
        plt.legend()
        plt.grid(True)

        if path is not None:
            plt.savefig(path)

        plt.show()
