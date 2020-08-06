import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Model Data Quality plugin | %(levelname)s - %(message)s')


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
        raise ValueError(
            'Unknown mode "{}". Please choose either "absolute_range" or "interquantile_range".'.format(mode))


def check_differences_between_datasets(df_ref, df_test, columns=None, range_mode='interquantile_range'):
    """
    :param df_ref:
    :param df_test:
    :param columns:
    :param range_mode:
        absolute_range returns the absolute min/max as bound
        interquantile_range use the quantiles: https://en.wikipedia.org/wiki/Interquartile_range
    :return: min/max bound
    """

    if not columns:
        columns = set(df_ref.columns).intersection(set(df_test.columns))
    else:
        columns = set(columns)

    features_not_in_dataset = list(set(columns) - set(df_test.columns))
    if len(features_not_in_dataset) > 0:
        raise ValueError("The following columns are used in the model but don't exist in this dataset: {}".format(' ,'.join(features_not_in_dataset)))

    logger.info('Columns to check: {}'.format(list(columns)))

    n_test = float(df_test.shape[0])
    # we concat the 2 df before checking the type of each column. This avoids situation where col A in ds_ref is object (1a,2,3) but in ds_test is numerical (4,5,6)
    concat_df = pd.concat([df_ref, df_test])

    numerical_columns_diff = {}
    categorical_columns_diff = {}

    for col_name in concat_df.select_dtypes(include=['number']):
        if col_name not in columns:
            continue
        lower_bound, upper_bound = get_bounds(df_ref[col_name], mode=range_mode)
        logger.info('Checking column {}. Lower bound: {}. Upper bound: {}'.format(col_name, lower_bound, upper_bound))
        # We count the ratio of samples in test outside of this range
        n_invalid = np.sum(np.logical_or(df_test[col_name].values < lower_bound, df_test[col_name].values > upper_bound))
        numerical_columns_diff[col_name] = n_invalid / n_test

    for col_name in concat_df.select_dtypes(include=['object', 'category']):
        if col_name not in columns:
            continue
        ref_categories = set(df_ref[col_name].unique())
        new_categories = set(df_test[col_name].unique())
        catogories_diff = list(new_categories - ref_categories)
        logger.info('Checking column {}. Reference categories: {}'.format(col_name, ref_categories))
        categorical_columns_diff[col_name] = catogories_diff

    return numerical_columns_diff, categorical_columns_diff
