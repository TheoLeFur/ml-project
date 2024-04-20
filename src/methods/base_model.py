import copy
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Tuple, Optional, Dict
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
    def set_hyperparameters(self, params: "Hyperparameters") -> None:
        """

        Args:
            params: Instance of Hyperparameters class containing the hyperparameters to tune on

        Returns: None

        """
        raise NotImplementedError

    def predict_and_tune(
            self,
            train_data: np.ndarray,
            ground_truth_train_labels: np.ndarray,
            params: List["Hyperparameters"],
            eval_criterion: Callable[[np.ndarray, np.ndarray], float],
            metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
            n_folds: Optional[int] = 1
    ) \
            -> Tuple[List[float], "Hyperparameters", float, Dict[str, float]]:

        """

        Args:
            train_data: Training data, of shape (N, D), where D is the number of featueres
            ground_truth_train_labels: Labels of shape (N, C), where C is the number of label features
            params: List of hyperparameters to evaluate on the model
            eval_criterion: Evaluation criterion used for the model. Used for determining the best hyperparameters
            metrics: Additional metrics one may want to use for evaluating the model. Defaulted to None.
            n_folds: Number of folds to use in cross-validation

        Returns: Loss for each hyperparameter value, the best hyperparameters, the best loss and the metrics.

        """
        min_value_train: float = np.Inf
        best_params: BaseModel.Hyperparameters = None
        best_logs: Dict[str, float] = {}
        train_losses: List = []

        for param in params:

            self.set_hyperparameters(param)

            train_loss, logs = self.predict_with_cv(
                train_data=train_data,
                train_labels=ground_truth_train_labels,
                n_folds=n_folds,
                criterion=eval_criterion,
                metrics=metrics
            )

            train_losses.append(train_loss)

            if train_loss < min_value_train:
                best_logs = copy.deepcopy(logs)
                min_value_train = train_loss
                best_params = copy.deepcopy(param)

        return train_losses, best_params, min_value_train, best_logs

    def predict_with_cv(
            self,
            train_data: np.ndarray,
            train_labels: np.ndarray,
            n_folds,
            criterion: Callable[[np.ndarray, np.ndarray], float],
            metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None) -> Tuple[
        float, Dict[str, float]]:

        """

        Args:
            metrics: A dictionary containing additional metrics on may want to apply during CV. A metric takes the
            form f: (dataset, labels) -> float
            criterion: Main criterion that will be used to perform hyperparameter selection
            train_data: training data, of shape (N,D)
            train_labels: training labels, of shape (N, C), where C is the number of features in labels
            n_folds: Number of folds to apply in cross-validation

        Returns: loss after cross-validation, other metrics

        """

        logs: Dict = {}
        if n_folds == 1:
            self.fit(train_data, train_labels)

            if metrics is not None:
                for key in metrics.keys():
                    logs[key] = 0
            train_labels_pred = self.predict(train_data)
            acc = criterion(train_labels_pred, train_labels)

            if metrics is not None:
                for key in metrics.keys():
                    metric: Callable[[np.ndarray, np.ndarray], float] = metrics[key]
                    val = metric(train_labels_pred, train_labels)
                    logs[key] = val
            return acc, logs

        else:

            if metrics is not None:
                for key in metrics.keys():
                    logs[key] = np.zeros(n_folds, dtype=np.float64)

            N = train_data.shape[0]
            accuracies: List[float] = []

            for i, fold in enumerate(range(n_folds)):

                indices = np.arange(N)
                split_size = N // n_folds

                validation_indices = indices[fold * split_size: (fold + 1) * split_size]
                train_indices = np.setdiff1d(indices, validation_indices, assume_unique=True)

                x_train_fold = train_data[train_indices, :]
                x_val_fold = train_data[validation_indices, :]
                if len(train_labels.shape) == 1:
                    y_train_fold = train_labels[train_indices]
                    y_val_fold = train_labels[validation_indices]
                else:
                    y_train_fold = train_labels[train_indices, :]
                    y_val_fold = train_labels[validation_indices, :]

                self.fit(x_train_fold, y_train_fold)
                val_pred_labels = self.predict(x_val_fold)
                acc = criterion(val_pred_labels, y_val_fold)
                accuracies.append(acc)

                if metrics is not None:
                    for key in metrics.keys():
                        metric: Callable[[np.ndarray, np.ndarray], float] = metrics[key]
                        val = metric(val_pred_labels, y_val_fold)
                        logs[key][i] = val

            return np.mean(accuracies), self.reduce_mean(copy.deepcopy(logs))

    @staticmethod
    def reduce_mean(hashmap: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Args:
            hashmap: A dictionary of the form (str, np.ndarray)

        Returns: A dictionary of the form (str, float), where the array has been reduced by averaging

        """
        reduced: Dict[str, float] = {}
        if len(hashmap) > 0:
            for key in hashmap.keys():
                reduced[key] = np.mean(hashmap[key])
        return reduced
