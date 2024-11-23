"""
Python file that collects functions related to a KNeighborsClassifier
model evaluation. In particular, the functions are:
- evaluate_classification, return accuracy score and log loss for a
    specific KNeighborsClassifier model.
- cv_bestK, returns a metrics DataFrame for models with different
    numbers of neighbours.
"""

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd


def evaluate_classification(mdl, dataset):
    """
    Function that returns both the accuracy score and the log
    loss of a classification model. To compute the log loss is
    necessary to use a model with the method predict_proba().

    Arguments:
    ---
    * mdl, classification model that have methods predict and
        predict_proba.
    * dataset (tuple), tuple of two elements that have as first
        argument the features array and as the second argument
        the target array.
    """
    X, y = dataset

    y_pred = mdl.predict(X)
    y_proba = mdl.predict_proba(X)
    
    return accuracy_score(y_true=y, y_pred=y_pred),\
        log_loss(y_true=y, y_pred=y_proba)

def cv_bestK(dataset, Ks, preprocessor, knn_weights='uniform', n_splits=5, rnd_seed=19, all_metrics=True, flag_balance_splits=True):
    """
    Function that execute the cross-validation of a specific
    dataset using the `sklearn.neighbors.KNeighborsClassifier`.
    It is necessary to specify the preprocessor to use to avoid
    issues of data leakage.

    Arguments:
    ---
    * dataset (tuple), tuple of two elements that have as first
        argument the features array and as the second argument
        the target array.
    * Ks (list), list with integer values of neighbours.
    * knn_weights (str, default=uniform), it represents the method
        the model gives weights to observation.
    * preprocessor, method to preprocess features. It must have
        methods `fit_transform` and `fit`.
    * n_splits (int), integer value that specify the number of
        splits for cross-validation.
    * rnd_seed (int), integer value to reproduce results.
    """

    X, y = dataset

    if flag_balance_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rnd_seed)
    else:
        # define splits object
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=rnd_seed)

    # define dictionary to store average metrics of cv
    metrics_dict={
        "n_neighbors":[],
        "train_accuracy_mean": [],
        "train_accuracy_std": [],
        "train_loss_mean": [],
        "train_loss_std": [],
        "test_accuracy_mean": [],
        "test_accuracy_std": [],
        "test_loss_mean": [],
        "test_loss_std": []
    }

    # create list to store the metrics for each fold
    if all_metrics:
        for fold_i in range(n_splits):
            metrics_dict[f"train_accuracy_{fold_i+1}"] = []
            metrics_dict[f"train_loss_{fold_i+1}"] = []
            metrics_dict[f"test_accuracy_{fold_i+1}"] = []
            metrics_dict[f"test_loss_{fold_i+1}"] = []

    for k in Ks:
        
        train_accuracy = []
        train_loss = []

        test_accuracy = []
        test_loss = []

        knn = KNeighborsClassifier(n_neighbors=k, weights=knn_weights)

        for fold_i, (train_index, test_index) in enumerate(splitter.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            knn.fit(X_train, y_train)

            tr_acc, tr_loss = evaluate_classification(mdl=knn,
                                                      dataset=(X_train, y_train))
            tst_acc, tst_loss = evaluate_classification(mdl=knn,
                                                        dataset=(X_test, y_test))
            
            train_accuracy.append(tr_acc)
            train_loss.append(tr_loss)
            test_accuracy.append(tst_acc)
            test_loss.append(tst_loss)
            if all_metrics:
                metrics_dict[f"train_accuracy_{fold_i+1}"].append(tr_acc)
                metrics_dict[f"train_loss_{fold_i+1}"].append(tr_loss)
                metrics_dict[f"test_accuracy_{fold_i+1}"].append(tst_acc)
                metrics_dict[f"test_loss_{fold_i+1}"].append(tst_loss)

        
        metrics_dict["n_neighbors"].append(k)
        metrics_dict["train_accuracy_mean"].append(np.array(train_accuracy).mean())
        metrics_dict["train_accuracy_std"].append(np.array(train_accuracy).std())
        metrics_dict["train_loss_mean"].append(np.array(train_loss).mean())
        metrics_dict["train_loss_std"].append(np.array(train_loss).std())
        metrics_dict["test_accuracy_mean"].append(np.array(test_accuracy).mean())
        metrics_dict["test_accuracy_std"].append(np.array(test_accuracy).std())
        metrics_dict["test_loss_mean"].append(np.array(test_loss).mean())
        metrics_dict["test_loss_std"].append(np.array(test_loss).std())

    return pd.DataFrame(metrics_dict)

def evaluate_knn(n_neighbors, train_ds, test_ds, knn_weight='uniform'):
    X_train, y_train = train_ds
    X_test, y_test = test_ds

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=knn_weight)
    knn.fit(X_train, y_train)
    knn_train_accuracy, knn_train_loss = evaluate_classification(
        knn,
        (X_train, y_train)
    )
    knn_test_accuracy, knn_test_loss = evaluate_classification(
        knn,
        (X_test, y_test)
    )
    print(f"\nNumber of neighbours: {n_neighbors}\n",
          f"\tTRAINING:\n\t- accuracy: {knn_train_accuracy:.2%}\n",
          f"\t- loss: {knn_train_loss:.4f}\n",
          f"\tTEST:\n\t- accuracy: {knn_test_accuracy:.2%}\n",
          f"\t- loss: {knn_test_loss:.4f}")
    return knn


def knn_best_K_over_entire_ds(train_ds, test_ds, Ks, knn_weight='uniform'):
    X_train, y_train = train_ds
    X_test, y_test = test_ds

    results_dict = {
        "n_neighbors": [],
        "train_accuracy_mean": [],
        "train_loss_mean": [],
        "test_accuracy_mean": [],
        "test_loss_mean": []
    }

    for k in Ks:
        knn = KNeighborsClassifier(n_neighbors=k, weights=knn_weight)
        knn.fit(X_train, y_train)
        
        train_accuracy, train_loss = evaluate_classification(
            knn,
            (X_train, y_train)
        )

        test_accuracy, test_loss = evaluate_classification(
            knn,
            (X_test, y_test)
        )

        results_dict["n_neighbors"].append(k)
        results_dict["train_accuracy_mean"].append(train_accuracy)
        results_dict["train_loss_mean"].append(train_loss)
        results_dict["test_accuracy_mean"].append(test_accuracy)
        results_dict["test_loss_mean"].append(test_loss)


    return pd.DataFrame(results_dict)