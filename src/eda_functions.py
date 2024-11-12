import pandas as pd

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