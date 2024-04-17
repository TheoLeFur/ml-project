import numpy as np
import sys
from ..utils import append_bias_term
from src.methods.base_model import BaseModel
from dataclasses import dataclass


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
        self.weight = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        # Append a bias term to the training data
        training_data_bias = append_bias_term(training_data)

        # Regularization matrix, excluding the bias term from regularization
        reg_matrix = self.lmda * np.eye(training_data_bias.shape[1])
        reg_matrix[-1, -1] = 0  # Do not regularize the bias term

        # Closed-form solution for weight calculation
        self.weights = np.linalg.inv(
            training_data_bias.T @ training_data_bias + reg_matrix) @ training_data_bias.T @ training_labels
        print(self.weights)
        print(self.weights.shape)
        # Return predictions for the training data to verify the fit
        pred_regression_targets = training_data_bias.dot(self.weights)
        return pred_regression_targets

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        # Append a bias term to the test data
        test_data_bias = append_bias_term(test_data)

        # Predict using the learned weights
        pred_regression_targets = test_data_bias.dot(self.weights)

        return pred_regression_targets
