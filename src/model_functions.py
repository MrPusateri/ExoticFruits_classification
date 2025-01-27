"""
Python file that collects functions related to a KNeighborsClassifier
model evaluation. In particular, the functions are:
- create_my_preprocessing, specific function for the project that
    return the specific processing ColumnTransformer.
- evaluate_classification, return accuracy score and log loss for a
    classification model.
- cv_bestK, returns a metrics DataFrame for models with different
    numbers of neighbours.
- custom_classification_cv, return accuracy and log loss for a
    classification model, applying cross-validation.
- evaluate_knn, return a KNN model for a specific number of neighbours
    and a method to weight points.
- apply_random_search, return best model after a random search CV.
- add_model_results, function to return DataFrame with specific for
    a classification model.
"""
import sklearn
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
import sklearn.pipeline

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

def create_my_preprocessing():
    """
    Function that create a specific preprocessing ColumnTransformer
    in order to create different objects with the same structure
    along the code and obtain different preprocessing without the
    knowledge of previous fitting.
    """
    # preprocess data
    impute_and_scale_col_index = [3]
    scale_col_index = [0, 1, 2, 4]

    # define a pipeline to impute missing values and normalize data
    impute_and_scale_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),
        ('scaler', MinMaxScaler())
    ])

    # build column transformer to preprocess features
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_scale', impute_and_scale_pipeline, impute_and_scale_col_index),
            ('scaler', MinMaxScaler(), scale_col_index)
        ],
        remainder='passthrough'
    )

    return preprocessor

def evaluate_classification(mdl, dataset, flag_show_report=False):
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
    * flag_show_report (bool, default=False), if True the
        function print on the standard output the classification
        report.
    """
    X, y = dataset

    y_pred = mdl.predict(X)
    y_proba = mdl.predict_proba(X)

    if flag_show_report:
        print(classification_report(y_true=y, y_pred=y_pred))
    
    return accuracy_score(y_true=y, y_pred=y_pred),\
        log_loss(y_true=y, y_pred=y_proba)

def cv_bestK(dataset, Ks, preprocessor, knn_weights='uniform', n_splits=5,
             rnd_seed=19, all_metrics=True, flag_balance_splits=True):
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
        knn = KNeighborsClassifier(n_neighbors=k, weights=knn_weights)

        train_accuracy, train_loss, test_accuracy, test_loss = custom_classification_cv(
            clf_mdl=knn,
            X=X,
            y=y,
            splitter=splitter,
            preprocessor=preprocessor)
        
        if all_metrics:
            for fold_i in range(n_splits):
                metrics_dict[f"train_accuracy_{fold_i+1}"].append(train_accuracy[fold_i])
                metrics_dict[f"train_loss_{fold_i+1}"].append(train_loss[fold_i])
                metrics_dict[f"test_accuracy_{fold_i+1}"].append(test_accuracy[fold_i])
                metrics_dict[f"test_loss_{fold_i+1}"].append(test_loss[fold_i])
        
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

def custom_classification_cv(clf_mdl, X, y, splitter, preprocessor):
    """
    Function that perform a cross-validation with a classification
    model. That allows to fit the preprocessing only on the
    training set. Moreover, the user can specify the splitter to
    use (KFold, StratifiedKFold, or whaterver).

    Arguments:
    ---
    * clf_mdl, classification model with methods `fit`, `predict`
        and `predict_proba`.
    * X, features array.
    * y, target array.
    * splitter, object with method `split(X, y)` that return index
        to split the dataset in training and test.
    * preprocessor, object with method `fit_transform` and
        `transform`.
    """    
    # list to save train metrics
    train_accuracy = []
    train_loss = []

    # list to save test metrics
    test_accuracy = []
    test_loss = []

    for train_index, test_index in splitter.split(X, y):
        # split training and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit preprocessor only on training data and tranform
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # fit classification model
        clf_mdl.fit(X_train, y_train)

        # evaluate performances
        tr_acc, tr_loss = evaluate_classification(mdl=clf_mdl,
                                                  dataset=(X_train, y_train))
        
        tst_acc, tst_loss = evaluate_classification(mdl=clf_mdl,
                                                    dataset=(X_test, y_test))
        
        train_accuracy.append(tr_acc)
        train_loss.append(tr_loss)
        test_accuracy.append(tst_acc)
        test_loss.append(tst_loss)
    
    return train_accuracy, train_loss, test_accuracy, test_loss

def evaluate_knn(n_neighbors, train_ds, test_ds, knn_weight='uniform', flag_print=True):
    """
    Function that train a `KNeighborsClassifier` with a specific
    number of neighbors and the possibility to specify the argument
    `weights`. The function print the accuracy on training and test
    set (if flag_print=True), and return the trained model.

    Arguments:
    ---
    * n_neighbors (int), number of neighbors.
    * train_ds (tuple), tuple of two array related to the training
        set. The first element is the features array and as the
        second one is the target array.
    * test_ds (tuple), tuple of two array related to the test
        set. The first element is the features array and as the
        second one is the target array.
    * knn_weight (default='uniform'), string that specify the method
        to adopt in order to assign weights to neighbors.
    * flag_print (default=True), flag to print the results.
    """
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

    if flag_print:
        print(f"\nNumber of neighbours: {n_neighbors}\n",
            f"\tTRAINING:\n\t- accuracy: {knn_train_accuracy:.2%}\n",
            f"\t- loss: {knn_train_loss:.4f}\n",
            f"\tTEST:\n\t- accuracy: {knn_test_accuracy:.2%}\n",
            f"\t- loss: {knn_test_loss:.4f}")
    return knn

def apply_random_search(
        params_dist: dict,
        model_pipeline: sklearn.pipeline.Pipeline,
        dataset: tuple,
        random_state: int=19,
        n_iter: int=100,
        n_splits: int=5,
        flag_balance_splits: bool=True,
        scoring: str='f1'):
    """
    Function to apply a random search with cross validation.
    In particular, it should be used with a Pipeline that contains
    the preprocessing of the features in to avoid data leakage
    issues.

    Arguments:
    ---
    * params_dist (dict), dictionary that allow to modify the
        parameters of the model in the pipeline used to solve the
        task.
    * model_pipeline (sklearn.pipeline.Pipeline), pipeline that
        has the preprocessing of the data and the model to solve
        the task.
    * dataset (tuple), tuple of two elements that have as first
        argument the features array and as the second argument
        the target array.
    * random_state (int, default=19), integer that represent the
        random seed in order to obtain reprudicible results.
    * n_iter (int, default=100), number of iteration in the
        random search.
    * n_splits (int, default=5), number of folds for the cross
        validation process.
    * flag_balance_splits (bool, default=True), if True the
        function to split the train and test set during the
        cross validation will split the data in order to have
        balanced classes (classification).
    * scoring='f1'
    """
    if flag_balance_splits:
        splitter = StratifiedKFold(n_splits=n_splits,
                                   shuffle=True,
                                   random_state=random_state)
    else:
        # define splits object
        splitter = KFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=random_state)
        
    X, y = dataset
    
    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=params_dist,
        n_iter=n_iter,
        cv=splitter,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    
    return random_search.best_estimator_

def add_model_results(result_df, clf_model, model_name, dataset_type,
                      train_ds, test_ds):
    """
    Function that add accuracy and loss for a classification model
    to a results DataFrame.

    Arguments:
    ---
    * result_df, dataframe where store results. 
    * clf_model, classification model to test.
    * model_name, name of the model.
    * dataset_type, name of the dataset used.
    * train_ds, (X_train, y_train)
    * test_ds (X_test, y_test)
    """
    
    train_accuracy, train_loss = evaluate_classification(clf_model,
                                                         train_ds)
    
    test_accuracy, test_loss = evaluate_classification(clf_model,
                                                       test_ds)
    
    if result_df.empty:
        return pd.DataFrame(data=[[model_name, dataset_type, train_accuracy,
                                  train_loss, test_accuracy, test_loss]],
                            columns=['model_name', 'dataset_type', 'train_accuracy',
                                     'train_loss', 'test_accuracy', 'test_loss'])
    else:
        new_df = pd.DataFrame(data=[[model_name, dataset_type, train_accuracy,
                                    train_loss, test_accuracy, test_loss]],
                              columns=['model_name', 'dataset_type', 'train_accuracy',
                                       'train_loss', 'test_accuracy', 'test_loss'])
        
        return pd.concat([result_df, new_df], axis=0).reset_index(drop=True)