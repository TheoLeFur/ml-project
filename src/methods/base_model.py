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
    def set_hyperparameters(self, params: "Hyperparameters"):
        raise NotImplementedError

    # TODO: make all methods accept other types of metrics. Pass the functions into a dictionary.
    def predict_and_tune(
            self,
            train_data: np.ndarray,
            ground_truth_train_labels: np.ndarray,
            params: List["Hyperparameters"],
            eval_criterion: Callable[[np.ndarray, np.ndarray], float],
            metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
            n_folds: Optional[int] = 1
    ) \
            -> Tuple[List[float], "Hyperparameters", float, Dict[str, float]]:

        """

        Args:
            metrics:
            n_folds:
            ground_truth_train_labels:
            train_data:
            params:
            eval_criterion: The criterion should induce the inverse order of maximality: criterion(x) < criterion(y)
            => y > x.

        Returns:

        """
        min_value_train: float = np.Inf
        best_params: BaseModel.Hyperparameters = None
        best_metrics: Dict[str, float] = {}
        train_losses: List = []

        for param in params:
            self.set_hyperparameters(param)

            train_loss, metrics = self.predict_with_cv(train_data=train_data, train_labels=ground_truth_train_labels,
                                                       n_folds=n_folds, criterion=eval_criterion, metrics=metrics)

            train_losses.append(train_loss)

            if train_loss < min_value_train:
                best_metrics = metrics
                min_value_train = train_loss
                best_params = copy.deepcopy(param)

        return train_losses, best_params, min_value_train, best_metrics

    def predict_with_cv(self, train_data: np.ndarray, train_labels: np.ndarray, n_folds,
                        criterion: Callable[[np.ndarray, np.ndarray], float],
                        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]]) -> Tuple[
        float, Dict[str, float]]:

        """

        Args:
            metrics:
            criterion:
            train_data: training data, of shape (N,D)
            train_labels: training labels, of shape (N, C), where C is the number of features in labels
            n_folds: Number of folds to apply in CV

        Returns:

        """

        logs: Dict = {}

        if n_folds == 1:

            for key in metrics.keys():
                logs[key] = 0

            train_labels_pred = self.predict(train_data)
            acc = criterion(train_labels_pred, train_labels)

            for key in metrics.keys():
                metric: Callable[[np.ndarray, np.ndarray], float] = metrics[key]
                val = metric(train_labels_pred, train_labels)
                logs[key] = val
            return acc, logs

        else:
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
                for key in metrics.keys():
                    metric: Callable[[np.ndarray, np.ndarray], float] = metrics[key]
                    val = metric(val_pred_labels, y_val_fold)
                    logs[key][i] = val

            return np.mean(accuracies), self.reduce_mean(logs)

    @staticmethod
    def reduce_mean(hashmap: Dict[str, np.ndarray]) -> Dict[str, float]:
        reduced: Dict[str, float] = {}
        for key in hashmap.keys():
            reduced[key] = np.mean(hashmap[key])

        return reduced
