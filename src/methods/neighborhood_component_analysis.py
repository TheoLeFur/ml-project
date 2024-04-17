import numpy as np
from typing import Optional, Tuple, Any

from tqdm import tqdm


class NeighborhoodComponentAnalysis:

    def __init__(
            self,
            n_dims: int,
            learning_rate: Optional[float] = 1e-3,
            max_iter: Optional[int] = 500,
            tol: Optional[float] = 1e-5
    ):
        """
        
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

    def fit(
            self,
            training_data: np.ndarray,
            training_labels: np.ndarray
    ):
        """

        Args:
            training_data:
            training_labels:

        Returns:

        """
        n_samples, n_features = training_data.shape
        self.training_data = training_data
        self.training_labels = training_labels
        self.W: np.ndarray = np.random.randn(n_features, self.n_dims)

        loss_hist = []

        for _ in tqdm(range(self.max_iter)):
            probabilities: np.ndarray = self._compute_probabilities(training_data)
            classification_probabilities: np.ndarray = self._compute_classification_probabilities(
                probabilities,
                training_data,
                training_labels)

            loss: float = np.log(classification_probabilities).sum()

            outer: np.ndarray = np.einsum(
                'ac,bd->abcd',
                training_data,
                training_data
            )

            grad_loss: np.ndarray = (probabilities[:, :, np.newaxis, np.newaxis] * outer).sum(axis=(0, 1))

            for i in range(training_data.shape[0]):
                class_indices: np.ndarray = self.get_class_indices(i, training_data, training_labels)
                outer_same_class: np.ndarray = outer[i, class_indices, :, :].squeeze(axis=0)
                sum: np.ndarray = np.multiply(probabilities[np.newaxis, i, class_indices], outer_same_class).squeeze()
                grad_loss -= sum / classification_probabilities[i]
            loss_hist.append(loss)
            grad_loss = 2 * grad_loss * self.W
            self.W += self.learning_rate * grad_loss

            print(f"Loss value: {loss} \n")

    def _compute_probabilities(self, training_data) -> np.ndarray:
        """

        Args:
            training_data:
            training_labels:

        Returns:

        """

        transformed_data: np.ndarray = np.dot(training_data, self.W)
        differences: np.ndarray = transformed_data[:, np.newaxis, :] - transformed_data[np.newaxis, :, :]
        distances: np.ndarray = np.exp(-np.sum(differences ** 2, axis=2))
        row_sums: np.ndarray = np.sum(distances, axis=1)
        res: np.ndarray = distances / row_sums[:, np.newaxis]

        return res

    def _compute_classification_probabilities(
            self,
            probabilities: np.ndarray,
            training_data: np.ndarray,
            training_labels: np.ndarray) -> np.ndarray:
        """

        Args:
            probabilities:
            training_data:
            training_labels:

        Returns:

        """

        N: int = training_data.shape[0]
        classification_probabilities: np.ndarray = np.zeros(N)

        for i in range(N):
            prob = probabilities[i, :]
            same_class_indices = self.get_class_indices(i, training_data, training_labels)
            classification_probabilities[i] = np.sum(prob[same_class_indices])

        return classification_probabilities

    def get_class_indices(self, i, training_data: np.ndarray, training_labels: np.ndarray) -> np.ndarray | float:
        """

        Args:
            i:
            training_data:
            training_labels:

        Returns:

        """
        if i < 0 or i >= training_data.shape[0]:
            raise IndexError("Index out of range")

        element_class = training_labels[i]
        same_class_indices = np.where(training_labels == element_class)

        return same_class_indices
