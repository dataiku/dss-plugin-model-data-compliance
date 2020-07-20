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
    df_ref = None
    df_test = ds_test.get_dataframe()

    if config.get('input_mode') == 'dataset':
        if config.get('ds_ref') is None:
            raise ValueError('Please choose a reference dataset')
        df_ref = dataiku.Dataset(config.get("ds_ref")).get_dataframe()
        columns = config.get("columns_dataset")
    else:
        if config.get('model_ref') is None:
            raise ValueError('Please choose a reference model')
        model_ref = config.get('model_ref')
        model = dataiku.Model(model_ref)
        model_handler = get_model_handler(model)
        model_accessor = ModelAccessor(model_handler)
        df_ref = model_accessor.get_original_test_df()
        selected_features = model_accessor.get_selected_features()
        chosen_columns = config.get("columns_model")
        if len(chosen_columns) > 0:
            columns = chosen_columns
            features_not_in_model = list(set(columns) - set(selected_features))
            if len(features_not_in_model) > 0:
                raise ValueError('The following chosen columns are not used in the model: {}'.format(features_not_in_model))
        else:
            columns = selected_features

    diff = check_differences_between_datasets(df_ref, df_test, columns=columns)

    message = ' '.join(['[{}, {:.2f}, {}]'.format(col_name, ratio, 'PASSED' if ratio <= tolerance else 'FAILED') for col_name, ratio in diff.items()])
    
    anomaly = np.any(np.asarray(diff.values()) > tolerance)
    result = 'OK'
    if anomaly:
        result = 'ERROR' if werror else 'WARNING'
    return (result, message)
