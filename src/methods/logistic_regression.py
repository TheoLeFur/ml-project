from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.methods.base_model import BaseModel
from ..utils import get_n_classes, label_to_onehot


class LogisticRegression(BaseModel):
    """
    Logistic regression classifier.
    """

    @dataclass
    class LRHyperparameters(BaseModel.Hyperparameters):
        lr: float

    def set_hyperparameters(self, params: "LRHyperparameters"):
        self.lr = params.lr

    def __init__(self, lr, max_iters=500, task_kind="classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations

        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # get number of classes i.e. number of unique labels
        num_classes = get_n_classes(training_labels)
        # get number of features i.e. number of parameters in training data
        num_features = training_data.shape[1]
        # initialize weights
        self.weights = np.zeros((num_classes, num_features))
        # convert labels to one-hot, shape (N, C) where C is the number of classes
        onehot_labels = label_to_onehot(training_labels, num_classes)

        # gradient descent
        for _ in tqdm(range(self.max_iters)):
            # compute gradients
            grad = self._gradient_logistic_multi(
                training_data, onehot_labels, self.weights.T
            )
            # update weights
            self.weights = self.weights - self.lr * grad.T

        # return predictions
        pred_labels = self._logistic_regression_predict_multi(training_data, self.weights)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        pred_labels = self._logistic_regression_predict_multi(test_data, self.weights)
        return pred_labels

    @staticmethod
    def f_softmax(data, W):
        """
        Softmax function for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        exp_z = np.exp(data @ W)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        elem = self.f_softmax(data, W) - labels
        grad = data.T @ elem
        return grad

    def _logistic_regression_predict_multi(self, data, W):
        """
        Prediction the label of data for multi-class logistic regression.

        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        return np.argmax(self.f_softmax(data, W.T), axis=1)
