import copy
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Tuple, Optional
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
            params: List["Hyperparameters"],
            eval_criterion: Callable[[np.ndarray, np.ndarray], float],
            n_folds: Optional[int] = 1
    ) \
            -> Tuple[List[float], "Hyperparameters", float]:

        """

        Args:
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
        train_losses: List = []

        for param in params:
            self.set_hyperparameters(param)

            train_loss = self.predict_with_cv(
                train_data=train_data,
                train_labels=ground_truth_train_labels,
                n_folds=n_folds,
                criterion=eval_criterion,

            )

            train_losses.append(train_loss)

            if train_loss < min_value_train:
                min_value_train = train_loss
                best_params = copy.deepcopy(param)

        print(
            f"The best params are: {best_params}."
            f" \n They achieve score {min_value_train} on train data.")

        return train_losses, best_params, min_value_train

    def predict_with_cv(self, train_data: np.ndarray, train_labels: np.ndarray, n_folds,
                        criterion: Callable[[np.ndarray, np.ndarray], float]):

        """

        Args:
            criterion:
            train_data: training data, of shape (N,D)
            train_labels: training labels, of shape (N, C), where C is the number of features in labels
            n_folds: Number of folds to apply in CV

        Returns:

        """

        if n_folds == 1:
            self.fit(train_data, train_labels)
            train_labels_pred = self.predict(train_data)
            acc = criterion(train_labels_pred, train_labels)

            return acc
        else:
            N = train_data.shape[0]
            accuracies: List[float] = []

            for fold in range(n_folds):

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

            return np.mean(accuracies)
