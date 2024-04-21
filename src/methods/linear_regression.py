
import numpy as np
import sys
from ..utils import append_bias_term
from src.methods.base_model import BaseModel
from dataclasses import dataclass

import numpy as np

from src.methods.base_model import BaseModel
from ..utils import append_bias_term


class LinearRegression(BaseModel):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    @dataclass
    class LRHyperparameters(BaseModel.Hyperparameters):
        lmda: float

    def set_hyperparameters(self, params: "LRHyperparameters"):
        self.lmda = params.lmda

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.weights = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        # Regularization matrix, excluding the bias term from regularization
        reg_matrix = self.lmda * np.eye(training_data.shape[1])
        reg_matrix[-1, -1] = 0  # Do not regularize the bias term

        # Closed-form solution for weight calculation
        self.weights = np.linalg.inv(
            training_data.T @ training_data + reg_matrix) @ training_data.T @ training_labels
        # Return predictions for the training data to verify the fit
        pred_regression_targets = training_data.dot(self.weights)
        return pred_regression_targets

    def fit_with_gradient_descent(self, training_data, training_labels, learning_rate=0.01, iterations=1000):
        """
        Trains the model using gradient descent, returns predicted labels for training data.
        """
        n_samples, n_features = training_data.shape
        n_targets = training_labels.shape[1] if training_labels.ndim > 1 else 1
    
        # Initialize weights as a 2D array if training_labels is 2D, else as a 1D array
        self.weights = np.zeros((n_features, n_targets)) if n_targets > 1 else np.zeros(n_features)
    
        for _ in range(iterations):
            predictions = training_data.dot(self.weights)
            errors = predictions - training_labels
            gradients = 2/n_samples * training_data.T.dot(errors) + self.lmda * self.weights
            self.weights -= learning_rate * gradients
    
        pred_regression_targets = training_data.dot(self.weights)
        return pred_regression_targets

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        # Predict using the learned weights
        pred_regression_targets = test_data.dot(self.weights)

        return pred_regression_targets
