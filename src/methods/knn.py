import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm
from typing import Dict, Callable

from src.methods.base_model import BaseModel
from dataclasses import dataclass

EPSILON = 1e-12


class KNN(BaseModel):
    """
        kNN classifier object.
    """

    @dataclass
    class KNNHyperparameters(BaseModel.Hyperparameters):
        k: int

    def set_hyperparameters(self, params: "KNNHyperparameters"):
        self.k = params.k

    def __init__(
            self,
            k: Optional[int] = 1,
            task_kind: Optional[str] = "classification",
            weights_type: Optional[str] = None
    ):
        """

        Args:
            k: Number of neighbors used in KNN
            task_kind: classification or regression, defaulted to classification
            weights_type: type of weights to use in distance computation, might be uniform or inverse_distance
        """
        self.k: int = k
        self.task_kind: str = task_kind
        self.training_data = None
        self.training_labels = None

        if weights_type is not None:
            self.weights_type = weights_type
        else:
            self.weights_type = "uniform"

    def fit(self, training_data, training_labels):
        """

            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data: np.ndarray = training_data
        self.training_labels: np.ndarray = training_labels

        return self.predict(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """

        n_test: int = test_data.shape[0]
        if len(self.training_labels.shape) == 1:
            test_labels: np.ndarray = np.empty((0, 1))
        else:
            test_labels: np.ndarray = np.empty((0, *self.training_labels.shape[1:]))

        for i in tqdm(range(n_test)):

            distances: np.ndarray = np.sqrt(np.sum(np.square(self.training_data - test_data[i]), axis=1))
            k_nearest_indices: np.ndarray = distances.argsort()[:self.k]
            nearest_k_labels: np.ndarray = self.training_labels[k_nearest_indices]

            if self.task_kind == "classification":
                label: np.ndarray = self.compute_classification_label(nearest_k_labels)
            elif self.task_kind == "regression":

                if self.weights_type == "inverse_distance":
                    nearest_k_distances: np.ndarray = distances[k_nearest_indices]
                    kwargs: Dict = {"distances": nearest_k_distances}
                    label: np.ndarray = self.compute_regression_label(nearest_k_labels, **kwargs)
                elif self.weights_type == "uniform":
                    label: np.ndarray = self.compute_regression_label(nearest_k_labels)
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError("KNN supports only classification or regression task types")

            test_labels = np.vstack([test_labels, label])
        return np.squeeze(test_labels)

    def compute_classification_label(self, k_nearest_labels: np.ndarray) -> np.ndarray:
        """

        Args:
            k_nearest_labels: Array that contains the k nearest labels of a given point

        Returns: The label assigned to the point, assuming a classification label

        """
        return np.argmax(np.bincount(k_nearest_labels)).reshape(-1, 1)

    def compute_regression_label(self, k_nearest_labels: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            k_nearest_labels: Array that contains the k nearest labels of a given point

        Returns: The label assigned to the point, assuming a regression label

        """
        if self.weights_type == "uniform":
            return np.mean(k_nearest_labels, axis=0)
        elif self.weights_type == "inverse_distance":
            distances: np.ndarray = kwargs["distances"]

            # Handle the case where d(x, xk) = 0 for inverse calculation:
            mask: np.ndarray = (distances == 0)
            distances[mask] = EPSILON
            inverse_distances: np.ndarray = 1 / distances
            return np.average(k_nearest_labels, axis=0, weights=inverse_distances)
        else:
            raise NotImplementedError(f"KNN with weights type {self.weights_type} is not implemented")
