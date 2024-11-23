import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA

def plot_train_test_confusion_matrix(clf_mdl, train_ds, test_ds, fig_number=None, labels=None):
    """
    Function to plot confusion matrix of training and test predictions.

    Arguments:
    * clf_mdl (estimator instance), Fitted classifier or a fitted Pipeline in which the
        last estimator is a classifier.
    * train_ds (tuple), where the first argument is the features array and the second
        is the target array related to the training.
    * test_ds (tuple), where the first argument is the features array and the second
        is the target array related to the test.
    * fig_number (int, default=None), the number of the figure to display.
    * labels (list), a list containing the names of the labels to display.
    """
    # retrieve training data
    X_train, y_train = train_ds
    
    # retrieve test data
    X_test, y_test = test_ds

    # create a figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                figsize=(13, 5), tight_layout=True)

    # plot training confusion matrix
    # create an object ConfusionMatrixDisplay for training
    ConfusionMatrixDisplay.from_estimator(estimator=clf_mdl,
                                          X=X_train,
                                          y=y_train,
                                          display_labels=labels,
                                          ax=ax1,
                                          cmap="Greens")
    ax1.set(title="(1) Train")

    # plot test confusion matrix

    # create an object ConfusionMatrixDisplay for test
    ConfusionMatrixDisplay.from_estimator(estimator=clf_mdl,
                                          X=X_test,
                                          y=y_test,
                                          display_labels=labels,
                                          ax=ax2,
                                          cmap="Greens")
    ax2.set(title="(2) Test")

    if fig_number:
        fig.text(0.5, -0.02,
                 f'Fig {fig_number}: Confusion matrix over training and test sets.',
                 ha='center',
                 fontsize=12)
    plt.show()

def show_cv_results(metrics_cv_df, min_neighbors, n_splits, fig_number=None):
    """
    Fuction to show average accuracy and log loss for knn with cross validation.

    Arguments:
    ---
    * metrics_cv_df (pandas.DataFrame), it must be a DataFrame with the same
        structure of the `cv_bestK` output.
    * min_neighbors (int), minimum number of neighbors to show.
    * n_splits (int), number of splits used in the cross-validation.
    * fig_number (int), number of the figure to generate.
    """
    mask = metrics_cv_df.n_neighbors>=min_neighbors

    # define best mean accuracy for the tests
    best_test_accuracy = metrics_cv_df.loc[mask, 'test_accuracy_mean'].max()
    best_k_accuracy = (
        metrics_cv_df
        .loc[
            metrics_cv_df.test_accuracy_mean==best_test_accuracy,
            'n_neighbors']
        .to_list()
    )

    # define best mean log loss for the tests
    best_test_loss = metrics_cv_df.loc[mask, 'test_loss_mean'].min()
    best_k_loss = (
        metrics_cv_df
        .loc[
            metrics_cv_df.test_loss_mean==best_test_loss,
            'n_neighbors']
        .to_list()
    )

    if n_splits>1:
        title_str = f"{n_splits}-fold cross-validation"
        fig_title = f"with a {n_splits}-Fold cross validation"
    else:
        title_str = "entire dataset"
        fig_title = title_str

    # plot figures
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                                   figsize=(10, 4*2),
                                   tight_layout=True)
    # ACCURACY
    ax1.plot(metrics_cv_df.loc[mask, 'n_neighbors'],
             metrics_cv_df.loc[mask, 'train_accuracy_mean'],
             label="Train mean")
    ax1.plot(metrics_cv_df.loc[mask, 'n_neighbors'],
             metrics_cv_df.loc[mask, 'test_accuracy_mean'],
             label="Test mean")
    ax1.scatter(best_k_accuracy,
                [best_test_accuracy]*len(best_k_accuracy),
                color='red',
                label="Best K")
    ax1.legend()
    ax1.grid()
    ax1.set(title=f"(1) Average accuracy over {title_str}\nBest Ks for test = {sorted(best_k_accuracy)}",
            xlabel="Number of neighbours",
            ylabel="Accuracy")

    # LOSS
    ax2.errorbar(metrics_cv_df.loc[mask, 'n_neighbors'],
                 metrics_cv_df.loc[mask, 'train_loss_mean'],
                 label="Train mean")
    ax2.errorbar(metrics_cv_df.loc[mask, 'n_neighbors'],
                 metrics_cv_df.loc[mask, 'test_loss_mean'],
                 label="Test mean")
    ax2.scatter(best_k_loss,
                [best_test_loss]*len(best_k_loss),
                color='red',
                label="Best K")
    ax2.legend()
    ax2.grid()
    ax2.set(title=f"(2) Average loss over {title_str}\nBest Ks for test = {sorted(best_k_loss)}",
            xlabel="Number of neighbours",
            ylabel="Log loss")

    # add figure number
    if fig_number:
        fig.text(0.5, -0.05,
                f'Fig {fig_number}: Best number of neighbours {fig_title}.',
                ha='center',
                fontsize=12)
    plt.show()

def show_cv_best_accuracy_box_plot(metrics_cv_df, min_neighbors, fig_number=None):
    """
    Function to plot box plot for the models with the best number of neighbors
    related to the average test accuracy.

    Arguments:
    ---
    * metrics_cv_df (pandas.DataFrame), it must be a DataFrame with the same
        structure of the `cv_bestK` output.
    * min_neighbors (int), minimum number of neighbors to show.
    * fig_number (int), number of the figure to generate.
    """
    mask = metrics_cv_df.n_neighbors>=min_neighbors
    best_test_accuracy = metrics_cv_df.loc[mask, 'test_accuracy_mean'].max()
    best_k_accuracy = (
        set(
            metrics_cv_df
            .loc[
                metrics_cv_df.test_accuracy_mean==best_test_accuracy,
                'n_neighbors']
            .to_list()
        )
    )

    # Drop mean information
    columns_to_drop = ["train_accuracy_mean", "train_loss_mean",
                       "test_accuracy_mean", "test_loss_mean",
                       "train_accuracy_std", "train_loss_std",
                       "test_accuracy_std", "test_loss_std"]
    best_accuracy_folds_df = metrics_cv_df.drop(columns=columns_to_drop)

    # get metrics related to Ks with best accuracy
    best_accuracy_folds_df = best_accuracy_folds_df[best_accuracy_folds_df.n_neighbors.isin(list(best_k_accuracy))]
    best_accuracy_folds_df = best_accuracy_folds_df.set_index("n_neighbors").T
    best_accuracy_folds_df = best_accuracy_folds_df[best_accuracy_folds_df.index.str.contains('accuracy')]

    # Show box plots
    fig, ax = plt.subplots(1,1, figsize=(7, 5), tight_layout=True)
    sns.boxplot(best_accuracy_folds_df, ax=ax, zorder=2)
    ax.set(xlabel="Number of neighbours",
        ylabel="Accuracy")
    ax.grid()
    if fig_number:
        fig.text(0.5, -0.02,
                 f'Fig {fig_number}: Box plots of the best number of neighbours related to accuracy.',
                 ha='center',
                 fontsize=12)
    plt.show()

def plot_3D_class(clf_mdl, dataset, decode_labels_dict, flag_train, fig_number):
    """
    Function to make a 3D scatter plot.
    """
    X, y = dataset
    y_pred = clf_mdl.predict(X)

    y_pred_decoded = np.array(list(map(lambda x: decode_labels_dict[x],
                                       y_pred)))
    y_decoded = np.array(list(map(lambda x: decode_labels_dict[x],
                                  y)))

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    pca_df = pd.concat(
        [
            pd.DataFrame(X_pca, columns=['component_1', 'component_2', 'component_3']),
            pd.Series(y_pred_decoded, name='predicted'),
            pd.Series(y_decoded, name='true')
        ],
        axis=1
    )
    pca_df.loc[:, 'class'] = pca_df.apply(
        lambda row: (
            row.true if row.predicted==row.true
            else 'Error'
        ),
        axis=1
    )

    fig = px.scatter_3d(pca_df,
                        x='component_1',
                        y='component_2',
                        z='component_3',
                        color='class')
    fig.show()
    if flag_train:
        print(f"Fig {fig_number}: 3D plot of differences between predicted and true classes over the training set.")
    else:
        print(f"Fig {fig_number}: 3D plot of differences between predicted and true classes over the test set.")
    