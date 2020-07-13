import numpy as np
import dataiku
from dku_model_data_quality.model_utils import check_differences_between_datasets
from dku_model_accessor import get_model_handler, ModelAccessor

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

    werror = config.get('werror')
    tolerance = config.get("tolerance")

    columns = None

    if config.get('input_mode') == 'dataset':
        ds_ref = dataiku.Dataset(config.get("ds_ref"))
        df_ref = ds_ref.get_dataframe()
    else:
        model_ref = config.get('model_ref')
        model = dataiku.Model(model_ref)
        model_handler = get_model_handler(model)
        model_accessor = ModelAccessor(model_handler)
        df_ref = model_accessor.get_original_test_df()
        columns = model_accessor.get_selected_features()

    select_columns = config.get("select_columns", False)
    if select_columns:
        columns = config.get("columns")

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
