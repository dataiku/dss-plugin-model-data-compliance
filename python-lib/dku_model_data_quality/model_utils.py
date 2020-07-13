import numpy as np


def check_differences_between_datasets(df_ref, df_test, columns=None):

    if not columns:
        columns = set(df_ref.columns).intersection(set(df_test.columns))
    else:
        columns = set(columns)
    
    result = dict()
    n_test = float(df_ref.shape[0])
    
    for col_name in df_test.select_dtypes(include=['number']):
        if not col_name in columns:
            continue
        rmin, rmax = df_ref[col_name].min(), df_ref[col_name].max()
        # We count the ratio of samples in test outside of this range
        n_invalid = np.sum(np.logical_or(df_test[col_name].values < rmin, df_test[col_name].values > rmax))
        result[col_name] = n_invalid / n_test

    for col_name in df_test.select_dtypes(include=['object', 'category']):
        if not col_name in columns:
            continue
        ref_categories = df_ref[col_name].unique()
        print(col_name, np.isin(df_test[col_name].values, ref_categories))
        result[col_name] = 1 - (np.isin(df_test[col_name].values, ref_categories).sum()) / n_test
    
    return result