import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import numpy as np


def ut_standard_col_name(input_df: pd.DataFrame):
    '''
    Function to standadize column names of a
    pandas DataFrame.

    Arguments:
    ---
    * input_df (pandas.DataFrame), DataFrame to
        standardise column names.
    '''
    column_names = {i: "_".join(i.split(" ")).lower() for i in input_df.columns}
    return input_df.rename(columns=column_names)