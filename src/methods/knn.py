import numpy as np
from typing import Optional
from tqdm import tqdm


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k: Optional[int] = 1, task_kind: Optional[str] = "classification"):
        """
            Call set_arguments function of this class.
        """

        self.k: int = k
        self.task_kind: str = task_kind
        self.training_data = None
        self.training_labels = None

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
                label: np.ndarray = self.compute_regression_label(nearest_k_labels)
            else:
                raise NotImplementedError("KNN supports only classification or regression task types")

            test_labels = np.vstack([test_labels, label])
        return np.squeeze(test_labels)

    @staticmethod
    def compute_classification_label(k_nearest_labels: np.ndarray) -> np.ndarray:
        """

        Args:
            k_nearest_labels: Array that contains the k nearest labels of a given point

        Returns: The label assigned to the point, assuming a classification label

        """
        return np.argmax(np.bincount(k_nearest_labels)).reshape(-1, 1)

    @staticmethod
    def compute_regression_label(k_nearest_labels: np.ndarray) -> np.ndarray:
        """
        Args:
            k_nearest_labels: Array that contains the k nearest labels of a given point

        Returns: The label assigned to the point, assuming a regression label

        """
        return np.mean(k_nearest_labels, axis=0)
