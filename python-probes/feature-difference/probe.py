from dku_model_data_quality.model_utils import check_differences_between_datasets
from dku_tools import get_params


def process(ds_test, partition_id):

    df_test = ds_test.get_dataframe()
    df_ref, columns, range_mode = get_params(config)
    diff = check_differences_between_datasets(df_ref, df_test, columns=columns, range_mode=range_mode)

    return dict([('Ratio of invalid samples in {}'.format(k), v) for k, v in diff.items()])