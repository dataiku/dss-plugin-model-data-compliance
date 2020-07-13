import dataiku
from dku_model_data_quality.model_utils import check_differences_between_datasets
from dku_model_accessor import get_model_handler, ModelAccessor


# Define here a function that returns the metric.
def process(ds_test, partition_id):
    # dataset is a dataiku.Dataset object
    
    # the values for the probe parameters are available as a dict
    # named config , and the plugin-level parameters (if any) as 
    # a dict named plugin_config

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

    return dict([('Ratio of invalid samples in {}'.format(k), v) for k, v in diff.items()])