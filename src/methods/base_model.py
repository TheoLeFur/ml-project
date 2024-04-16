import copy
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Tuple
from dataclasses import dataclass

import numpy as np


class BaseModel(metaclass=ABCMeta):
    @dataclass
    class Hyperparameters:
        pass

    @abstractmethod
    def fit(self, training_data, training_labels):
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data):
        raise NotImplementedError

    @abstractmethod
    def set_hyperparameters(self, params: "Hyperparameters"):
        raise NotImplementedError

    @abstractmethod
    def predict_and_tune(
            self,
            train_data: np.ndarray,
            ground_truth_train_labels: np.ndarray,
            test_data: np.ndarray,
            ground_truth_test_labels: np.ndarray,
            params: List["Hyperparameters"],
            eval_criterion: Callable[[np.ndarray, np.ndarray], float]) \
            -> Tuple[List[float], List[float], "Hyperparameters", float, float]:

        """

        Args:
            ground_truth_train_labels:
            train_data:
            test_data:
            ground_truth_test_labels:
            params:
            eval_criterion: The criterion should induce the inverse order of maximality: criterion(x) < criterion(y)
            => y > x.

        Returns:

        """
        min_value_test: float = np.Inf
        min_value_train: float = np.Inf

        best_params: BaseModel.Hyperparameters = None

        test_loss: List = []
        train_loss: List = []

        for param in params:
            self.set_hyperparameters(param)

            train_labels = self.predict(train_data)
            test_labels = self.predict(test_data)

            test_criterion = eval_criterion(test_labels, ground_truth_test_labels)
            train_criterion = eval_criterion(train_labels, ground_truth_train_labels)

            test_loss.append(test_criterion)
            train_loss.append(train_criterion)

            if test_criterion < min_value_test:

                min_value_train = train_criterion
                min_value_test = test_criterion
                best_params = copy.deepcopy(param)

        print(
            f"The best params are: {best_params}. \n They achieve score {min_value_test} on test data."
            f" \n They achieve score {min_value_train} on train data.")

        return train_loss, test_loss, best_params, min_value_train, min_value_test
