import numpy as np
from dku_model_data_quality.model_utils import check_differences_between_datasets
from dku_tools import get_params


def process(last_values, ds_test, partition_id):
    ds_test.add_read_partitions(partition_id)
    df_test = ds_test.get_dataframe()
    message_type = config.get('message_type', 'ERROR')
    tolerance = config.get("tolerance", 0)

    if (tolerance < 0) or (tolerance > 1):
        raise ValueError('Tolerance rate must between 0 and 1')

    df_ref, columns, range_mode = get_params(config)
    numerical_columns_diff, categorical_columns_diff = check_differences_between_datasets(df_ref, df_test, columns=columns, range_mode=range_mode)

    non_compliant_numerical_columns = []
    numerical_columns_anomaly = False
    for col_name, ratio in numerical_columns_diff.items():
        if ratio > tolerance:
            numerical_columns_anomaly = True
            non_compliant_numerical_columns.append('{} - {:.2f}'.format(col_name, ratio*100))

    if len(non_compliant_numerical_columns) > 0:
        numerical_columns_message = 'Numerical columns: {}% non-compliant samples'.format(', '.join(non_compliant_numerical_columns))
    else:
        numerical_columns_message = ''

    non_compliant_categorical_columns = []
    categorical_columns_anomaly = False
    for col_name, new_categories in categorical_columns_diff.items():
        if len(new_categories) > 0:
            categorical_columns_anomaly = True
            non_compliant_categorical_columns.append('{} - {} new categories'.format(col_name, len(new_categories)))

    if len(non_compliant_categorical_columns) > 0:
        categorical_columns_message = 'Categorical columns: {}'.format(', '.join(non_compliant_categorical_columns))
    else:
        categorical_columns_message = ''

    anomaly = numerical_columns_anomaly or categorical_columns_anomaly
    result = message_type if anomaly else'OK'
    message = '{}. {}'.format(numerical_columns_message, categorical_columns_message)

    return result, message
