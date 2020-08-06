from dku_model_data_quality.model_utils import check_differences_between_datasets
from dku_tools import get_params


def process(ds_test, partition_id):
    ds_test.add_read_partitions(partition_id)
    df_test = ds_test.get_dataframe()
    df_ref, columns, range_mode = get_params(config)
    numerical_columns_diff, diff_categorical_columns = check_differences_between_datasets(df_ref, df_test, columns=columns, range_mode=range_mode)

    numerical_columns_result = [('Ratio of invalid samples in {}'.format(k), v) for k, v in numerical_columns_diff.items()]
    categorical_columns_result = [('New categories in {}'.format(k), v) for k, v in diff_categorical_columns.items()]

    result = numerical_columns_result
    result.extend((categorical_columns_result))

    return dict(result)
