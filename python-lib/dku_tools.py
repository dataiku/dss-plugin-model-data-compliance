# -*- coding: utf-8 -*-
import dataiku

from dku_model_accessor import get_model_handler, ModelAccessor


def get_params(config):
    range_mode = config.get('range_mode')

    if config.get('input_mode') == 'dataset':
        df_ref = dataiku.Dataset(config.get("ds_ref")).get_dataframe(bool_as_str=True)
        columns = [col for col in config.get("columns_dataset") if col != '']
        columns_not_in_df_ref = set(columns) - set(df_ref.columns)
        if len(columns_not_in_df_ref) > 0:
            raise ValueError(
                'The following chosen columns are not in the reference dataset: {}. Please remove them from the list of columns to check.'.format(' ,'.join(list(columns_not_in_df_ref))))
    else:
        model_ref = config.get('model_ref')
        if model_ref is None:
            raise ValueError('Please choose a reference model.')
        model = dataiku.Model(model_ref)
        model_handler = get_model_handler(model)
        model_accessor = ModelAccessor(model_handler)
        df_ref = model_accessor.get_train_df()
        selected_features = model_accessor.get_selected_features()
        chosen_columns = [col for col in config.get("columns_model") if col != '']
        if len(chosen_columns) > 0:
            columns = chosen_columns
            features_not_in_model = list(set(columns) - set(selected_features))
            if len(features_not_in_model) > 0:
                raise ValueError('The following chosen columns are not used in the model: {}. Please remove them from the list of columns to check.'.format(' ,'.join(features_not_in_model)))
        else:
            columns = selected_features

    return df_ref, columns, range_mode
