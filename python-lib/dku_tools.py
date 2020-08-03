# -*- coding: utf-8 -*-
import dataiku
from dku_model_accessor import get_model_handler, ModelAccessor

def get_params(config):
    columns = None
    df_ref = None

    if config.get('input_mode') == 'dataset':
        df_ref = dataiku.Dataset(config.get("ds_ref")).get_dataframe()
        columns = [col for col in  config.get("columns_dataset") if col != '']
    else:
        model_ref = config.get('model_ref')
        model = dataiku.Model(model_ref)
        model_handler = get_model_handler(model)
        model_accessor = ModelAccessor(model_handler)
        df_ref = model_accessor.get_original_test_df()
        selected_features = model_accessor.get_selected_features()
        chosen_columns = [col for col in  config.get("columns_dataset") if col != '']#config.get("columns_model")
        if len(chosen_columns) > 0:
            columns = chosen_columns
            features_not_in_model = list(set(columns) - set(selected_features))
            if len(features_not_in_model) > 0:
                raise ValueError('The following chosen columns are not used in the model: {}. Please remove them from the list of columns to check'.format(
                        features_not_in_model))
        else:
            columns = selected_features

    range_mode = config.get('range_mode')

    return df_ref, columns, range_mode
