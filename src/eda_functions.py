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

def draw_mean_mode(x, **kwargs):
    plt.axvline(x.mean(), c='k', ls='--', lw=2.5, label='mean')
    plt.axvline(x.median(), c='orange', ls='--', lw=2.5, label='mode')
    
def show_dist_mean_mode(df, x, col):
    g = sns.displot(
        data=df,
        x=x,
        col=col,
        facet_kws=dict(sharey=False, sharex=False) # to have an x and y for each plot
    )
    g.map(draw_mean_mode, x)
    g.add_legend()
    return g