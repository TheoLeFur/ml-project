from typing import Optional

import numpy as np
from tqdm import tqdm


class ModelNotFitError(Exception):
    pass


class NeighborhoodComponentAnalysis:

    def __init__(
            self,
            n_dims: int,
            learning_rate: Optional[float] = 1e-3,
            max_iter: Optional[int] = 100,
            tol: Optional[float] = 1e-5,
            batch_size=128
    ):
        """
        Simple implementation of neighborhood component analysis. Ref: https://www.cs.toronto.edu/~hinton/absps/nca.pdf.

        Args:
            n_dims:
            learning_rate:
            max_iter:
            tol:
        """
        self.training_labels = None
        self.training_data = None
        self.W = None

        self.n_dims = n_dims
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size

    @property
    def transformation_matrix(self):
        if self.W is None:
            raise ModelNotFitError
        return self.W

    @property
    def n_components(self):
        return self.n_dims

    def fit(
            self,
            training_data: np.ndarray,
            training_labels: np.ndarray
    ):
        """

        Args:
            training_data: array of shape (N,D)
            training_labels: array of shape (N,)

        Returns:

        """
        n_samples, n_features = training_data.shape
        self.training_data = training_data
        self.training_labels = training_labels
        self.W = np.random.randn(n_features, self.n_dims)

        loss_hist = []
        outer = np.einsum(
            'ac,bd->abcd',
            training_data,
            training_data
        )
        for _ in tqdm(range(self.max_iter)):
            # For stochastic gradient descent
            idx = np.random.randint(0, training_data.shape[0], self.batch_size)
            training_data = training_data[idx]
            training_labels = training_labels[idx]
            outer = outer[idx, :, :, :][:, idx, :, :]

            probs = self._compute_probabilities(training_data)
            class_probs = self._compute_classification_probabilities(probs, training_labels)
            loss = np.sum(np.log(class_probs))

            mask = (training_labels[:, None] == training_labels[None, :])
            masked_probs = probs * mask
            grad_loss = 2 * ((probs[:, :, np.newaxis, np.newaxis] -
                              masked_probs[:, :, np.newaxis, np.newaxis]) * outer).sum(axis=(0, 1)) @ self.W

            self.W += self.learning_rate * grad_loss
            loss_hist.append(loss)
            print(f"Loss value: {loss}")

    def _compute_probabilities(self, training_data: np.ndarray) -> np.ndarray:
        """

        Args:
            training_data: array of shape (N,D)
            training_labels: array of shape (N,)

        Returns:

        """

        transformed_data = training_data @ self.W

        distances = self.pairwise_distances(transformed_data)
        res = np.exp(-distances)
        norm = np.sum(res, axis=1)
        return res / norm

    def _compute_classification_probabilities(
            self,
            probabilities: np.ndarray,
            training_labels: np.ndarray) -> np.ndarray:
        """

        Args:
            probabilities: Matrix of shape (N, N), where M[i,j] represents the probability of point i being classified as
            point j
            training_labels: array of shape (N,)

        Returns:

        """

        mask = (training_labels[:, None] == training_labels[None, :])
        classification_probabilities = np.sum((probabilities * mask), axis=1)

        return classification_probabilities

    def pairwise_distances(self, training_data: np.ndarray) -> np.ndarray:
        """

        Args:
            training_data: array of shape (N,D)

        Returns:

        """
        dot = np.matmul(training_data, training_data.T)
        norm_sq = np.diag(dot)
        dist = norm_sq[None, :] - 2 * dot + norm_sq[:, None]
        dist = np.clip(dist, a_min=0, a_max=None)
        return dist
