import numpy as np
import dataiku
from modelassertions.model_utils import check_differences_between_datasets


# This file is the actual code for the custom Python check check-against-reference

# Define here a function that returns the outcome of the check.
def process(last_values, ds_test, partition_id):
    # last_values is a dict of the latest values of the metrics,
    # with the values as a dataiku.metrics.MetricDataPoint.

    # Note that these are the 'current' values when running the check.
    # If the check runs after a build, you will get the values computed
    # after the build.

    # dataset is a dataiku.Dataset object.
    # You can use this object and the metrics API on it to access
    # history values

    # the values for the check parameters are available as a dict
    # named config , and the plugin-level parameters (if any) as 
    # a dict named plugin_config
    
    ds_ref = dataiku.Dataset(config.get("ds_ref"))
    select_columns = config.get("select_columns", False)
    columns = None
    if select_columns:
        columns = config.get("columns")
    werror = config.get("werror")
    tolerance = config.get("tolerance")
    
    df_ref = ds_ref.get_dataframe()    
    df_test = ds_test.get_dataframe()
    
    diff = check_differences_between_datasets(df_ref, df_test, columns=columns)
    
    message = ' '.join(['[{}, {}, {:.2f}]'.format(
        col_name, 'OK' if ratio <= tolerance else 'KO', ratio)
                        for col_name, ratio in diff.items()])
    
    anomaly = np.any(np.asarray(diff.values()) > tolerance)
    result = 'OK'
    if anomaly:
        result = 'ERROR' if werror else 'WARNING'
    return (result, message)
