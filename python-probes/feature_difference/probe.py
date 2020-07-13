# This file is the actual code for the custom Python probe feature_difference

from dataiku.metric import MetricDataTypes
import dataiku
from modelassertions.model_utils import check_differences_between_datasets


# Define here a function that returns the metric.
def process(ds_test, partition_id):
    # dataset is a dataiku.Dataset object
    
    # the values for the probe parameters are available as a dict
    # named config , and the plugin-level parameters (if any) as 
    # a dict named plugin_config
    
    ds_ref = dataiku.Dataset(config.get("ds_ref"))
    select_columns = config.get("select_columns", False)
    columns = None
    if select_columns:
        columns = config.get("columns")
    
    df_ref = ds_ref.get_dataframe()    
    df_test = ds_test.get_dataframe()
    
    diff = check_differences_between_datasets(df_ref, df_test, columns=columns)

    return dict([('Ratio of invalid samples in {}'.format(k), v) for k, v in diff.items()])