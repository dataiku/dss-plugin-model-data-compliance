import numpy as np
import dataiku
from dku_tools import get_params
from dku_model_data_quality.model_utils import check_differences_between_datasets

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


    df_test = ds_test.get_dataframe()
    werror = config.get('werror')
    tolerance = config.get("tolerance")
    df_ref, columns, range_mode = get_params(config)
    diff = check_differences_between_datasets(df_ref, df_test, columns=columns, range_mode=range_mode)

    message = ' '.join(['[{}, {:.2f}, {}]'.format(col_name, ratio, 'PASSED' if ratio <= tolerance else 'FAILED') for col_name, ratio in diff.items()])
    anomaly = np.any(np.asarray(diff.values()) > tolerance)
    result = 'OK'
    if anomaly:
        result = 'ERROR' if werror else 'WARNING'
    return (result, message)
