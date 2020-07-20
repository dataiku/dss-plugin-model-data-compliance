import dataiku
from dku_model_data_quality.model_utils import check_differences_between_datasets
from dku_tools import get_params

# Define here a function that returns the metric.
def process(ds_test, partition_id):
    # dataset is a dataiku.Dataset object
    
    # the values for the probe parameters are available as a dict
    # named config , and the plugin-level parameters (if any) as 
    # a dict named plugin_config

    """
    columns = None
    df_ref = None

    if config.get('input_mode') == 'dataset':
        df_ref = dataiku.Dataset(config.get("ds_ref")).get_dataframe()
        columns = config.get("columns_dataset")
    else:
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
                raise ValueError('The following chosen columns are not used in the model: {}. Please remove them from the list of columns to check'.format(features_not_in_model))
        else:
            columns = selected_features

    range_mode = config.get('range_mode')
    """
    df_test = ds_test.get_dataframe()
    df_ref, columns, range_mode = get_params(config)
    diff = check_differences_between_datasets(df_ref, df_test, columns=columns, range_mode=range_mode)

    return dict([('Ratio of invalid samples in {}'.format(k), v) for k, v in diff.items()])