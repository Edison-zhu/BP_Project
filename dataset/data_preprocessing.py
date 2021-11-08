import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer


def load_numerical_data(arff_paths_dict, dataset_name):
    data, meta = arff.loadarff(arff_paths_dict[dataset_name])
    df = pd.DataFrame(data)
    return df, meta


def drop_duplicates(df):
    df = df.drop_duplicates()
    print('There are {} duplicate items get dropped')
    return df


def get_null_counts(df):  # count all the null values inå df and null values in each column
    df_null_counts = {}
    df_null_all = df.isnull().values().sum ()
    df_null_counts['all'] = df_null_all
    for col in df.columns:
        df_null_counts[col] = df[col].isnull().values().sum()

    return df_null_counts


def drop_na(df, how='any', thresh=None):
    if how == 'any':
        df = df.dropna(how=how)
    if how == 'all':  # Passing how='all' will only drop rows that are all NA
        df = df.dropna(how=how)
    if how == 'thresh':
        thresh = int(input("You must input an integer for setting the threshold "))
        df = df.dropna(thresh=thresh)
    return df


def fill_na(df, fill_value=False, inplace=True, method=''):
    """"
    method: filling NA method
            ffill: an interpolation methods available for reindexing can be used with fillna
            mean: filling the missing value by mean value
    """
    if fill_value:
        value = float(input("You need to input a numerical specific value for NA"))
        df = df.fillna(value, inplace=inplace)
    else:
        if method is None:
            raise ValueError('You must define the fillna method if the fill_value is not given')
        elif method == 'mean':
            df_col_mean = df.mean()
            df = df.fillna(df_col_mean)
        elif method == 'ffill':
            df = df.fillna(method=method, limit=2)

    return df


def normalize_data(df, normalize_method='MinMax'):
    """
    There are three ways in sklearn to do normalization in our data:
    1. scale: Standardisation replaces the values by their Z scores
    2. StandardScale: This distribution will have values between -1 and 1with μ=0
    3. MinMaxScaler: This scaling brings the value between 0  and 1.
    4. Normalizer:
    """

    if normalize_method == 'MinMax':
        normalizer = MinMaxScaler
    elif normalize_method == 'Standard':
        normalizer = scale  # Standardisation replaces the values by their Z scores
    elif normalize_method == 'Mean':    # This distribution will have values between -1 and 1with μ=0
        normalize_method == StandardScaler
    elif normalize_method == 'UnitVector':
        normalizer = Normalizer

    normalize_df = df.apply(normalizer)  # Normalize data in each columns according to normalize_method

    return normalize_df
#
#
# def truncated_value(df, max_value, min_value):
#     truncated_df = None
#     return truncated_df
