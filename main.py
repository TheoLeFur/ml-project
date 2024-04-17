import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, compute_std, compute_means
import os
import torch
from typing import Dict
import matplotlib.pyplot as plt
from src.plotlib import PlotLib

np.random.seed(100)

task_name_to_task_type: Dict = {"center_locating": "regression",
                                "breed_identifying": "classification"}

# For GPU usage in the second part
device = "mps" if torch.backends.mps.is_available() else "cpu"


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('data/ms1/features.npz', allow_pickle=True)
        print(feature_data)
        xtrain, xtest, ytrain, ytest, ctrain, ctest = feature_data['xtrain'], feature_data['xtest'], \
            feature_data['ytrain'], feature_data['ytest'], feature_data['ctrain'], feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, 'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    else:
        raise NotImplementedError

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    xtrain = normalize_fn(
        data=xtrain,
        means=compute_means(xtrain),
        stds=compute_std(xtrain)
    )
    xtest = normalize_fn(
        data=xtest,
        means=compute_means(xtest),
        stds=compute_std(xtest)
    )

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        pass

    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(args.K, task_kind=task_name_to_task_type[args.task], weights_type="inverse_distance")
        ks: np.ndarray = np.arange(args.Kmin, args.Kmax, 1)
        params_list = []
        for k in ks:
            params_list.append(KNN.KNNHyperparameters(k=k))

    else:
        raise NotImplementedError

    ## 4. Train and evaluate the method
    if args.task == "center_locating":
        # Fit parameters on training data
        train_losses, best_params, best_train_loss = method_obj.predict_and_tune(
            xtrain,
            ctrain,
            params_list,
            mse_fn,
            n_folds=args.n_folds)

        PlotLib.plot_loss_against_hyperparam_val(list(map(lambda t: t.k, params_list)), train_losses)

        # Evaluation on test set
        method_obj.set_hyperparameters(best_params)
        method_obj.fit(xtrain, ctrain)
        test_labels = method_obj.predict(xtest)
        best_test_loss = mse_fn(test_labels, ctest)

        print(f"\nTrain loss = {best_train_loss:.3f}% - Test loss = {best_test_loss:.3f}")

    elif args.task == "breed_identifying":
        # Fit (:=train) the method on the training data for classification task. Since we measure hyperparameter
        # value in terms of the lowest loss, we take the opposite of the accuracy
        train_losses, best_params, best_train_loss = method_obj.predict_and_tune(
            xtrain,
            ytrain,
            params_list,
            lambda t, s: -accuracy_fn(t, s),
            n_folds=args.n_folds
        )

        print(f"\nTrain set: accuracy = {-best_train_loss:.3f}%")

        method_obj.set_hyperparameters(best_params)
        method_obj.fit(xtrain, ytrain)

        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)

        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        PlotLib.plot_loss_against_hyperparam_val(list(map(lambda t: t.k, params_list)), train_losses)
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    data = xtrain
    # Plot feature values:
    PlotLib.plot_feature_values(data)
    # Plot feature distribution
    PlotLib.plot_feature_distribution(data)


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str,
                        help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    # For hyperparameter search
    parser.add_argument('--Kmin', type=int, default=1, help="Minimum number of neighboring datapoints used for knn")
    parser.add_argument('--Kmax', type=int, default=1, help="Maximum number of neighboring datapoints used for knn")
    parser.add_argument('--lmdaMin', type=float, default=10, help="minimum lambda of linear/ridge regression")
    parser.add_argument('--lmdaMax', type=float, default=10, help="maximum lambda of linear/ridge regression")
    parser.add_argument('--lrmin', type=float, default=10, help='minimum learning rate')
    parser.add_argument('--lrmax', type=float, default=10, help='maximum learning rate')

    parser.add_argument('--n_folds', type=int, default=1, help="number of folds to use in cross validation")

    # Feel free to add more arguments here if you need!
    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
