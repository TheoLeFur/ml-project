import numpy as np
from typing import Optional, Callable, Dict

KNNReduction = Callable[[np.ndarray], np.ndarray]
str_to_reduction: Dict = {
    "classification": np.mean,
    "regression": lambda t: np.argmax(np.bincount(t))
}


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

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

        self.training_data = training_data
        self.training_labels = training_labels

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
        test_labels: np.ndarray = np.zeros(n_test)

        for i in range(n_test):

            distances: np.ndarray = np.sqrt(np.sum(np.square(self.training_data - test_data[i]), axis=1))
            k_nearest_indices: np.ndarray = distances.argsort()[:self.k]
            nearest_k_labels: np.ndarray = self.training_labels[k_nearest_indices]
            if isinstance(self.task_kind, str):
                test_labels[i] = nearest_k_labels[self.task_kind]
            else:
                raise TypeError(f"task kind should be str")
        return test_labels
