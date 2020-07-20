import numpy as np

def get_bounds(a, mode='absolute_range'):
    """

    :param a: array_like
    :param mode: absolute or quantile
    :return: lower bound, upper bound
    """
    if mode == 'absolute_range':
        return np.min(a), np.max(a)
    elif mode == 'interquantile_range':
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        iqr = upper_quartile - lower_quartile

        upper_bound = upper_quartile + 1.5 * iqr
        lower_bound = lower_quartile - 1.5 * iqr
        return lower_bound, upper_bound
    else:
        raise ValueError('Unknown mode "{}". Please choose either "absolute_range" or "interquantile_range".'.format(mode))


def check_differences_between_datasets(df_ref, df_test, columns=None, range_mode='interquantile_range'):

    if not columns:
        columns = set(df_ref.columns).intersection(set(df_test.columns))
    else:
        columns = set(columns)
    
    result = dict()
    n_test = float(df_ref.shape[0])
    
    for col_name in df_test.select_dtypes(include=['number']):
        if not col_name in columns:
            continue
        rmin, rmax = get_bounds(df_ref[col_name], mode=range_mode)
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