"""
This python file containts functions that could be useful to
explore data at the starting point of a project.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_info(df: pd.DataFrame):
    """
    Function that return a pandas.DataFrame object with
    some statistical information to have an initial
    analysis of the input DataFrame.

    Arguments:
    ---
    * df (pandas.DataFrame), DataFrame to analyse.
    """
    return (
        pd.concat(
            [
                df.dtypes.rename("dtype"),
                df.nunique().rename("n_unique"),
                df.isna().sum().rename("nan_values"),
                df.describe().transpose()
            ],
            axis=1
        )
    )

def ut_standard_col_name(input_df: pd.DataFrame):
    '''
    Function to standardize column names of a
    pandas DataFrame.

    Arguments:
    ---
    * input_df (pandas.DataFrame), DataFrame to
        standardise column names.
    '''
    column_names = {i: "_".join(i.split(" ")).lower() for i in input_df.columns}
    return input_df.rename(columns=column_names)

def __draw_mean_mode(x, **kwargs):
    plt.axvline(x.mean(), c='k', ls='--', lw=2.5, label='mean')
    plt.axvline(x.median(), c='orange', ls='--', lw=2.5, label='mode')
    
def show_dist_mean_mode(df, x, col):
    """
    Function to plot a distribution divided by possible values of `col` variable.
    Moreober, the graph show mean and mode value for the variable.

    Arguments:
    ---
    * df (pandas.core.DataFrame), DataFrame with the variable of interesting.
    * x (str), name of the variable to plot in the x-axis.
    * col (str), name of the variable with different classes to split the data.
    """
    g = sns.displot(
        data=df,
        x=x,
        col=col,
        facet_kws=dict(sharey=False, sharex=False) # to have an x and y for each plot
    )
    g.map(__draw_mean_mode, x)
    g.add_legend()
    return g