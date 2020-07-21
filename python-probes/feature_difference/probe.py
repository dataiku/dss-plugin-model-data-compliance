import dataiku
from dku_model_data_quality.model_utils import check_differences_between_datasets
from dku_tools import get_params

# Define here a function that returns the metric.
def process(ds_test, partition_id):
    # dataset is a dataiku.Dataset object
    
    # the values for the probe parameters are available as a dict
    # named config , and the plugin-level parameters (if any) as 
    # a dict named plugin_config

    df_test = ds_test.get_dataframe()
    df_ref, columns, range_mode = get_params(config)
    diff = check_differences_between_datasets(df_ref, df_test, columns=columns, range_mode=range_mode)

    return dict([('Ratio of invalid samples in {}'.format(k), v) for k, v in diff.items()])