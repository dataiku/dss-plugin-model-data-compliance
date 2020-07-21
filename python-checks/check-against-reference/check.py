import numpy as np
from dku_tools import get_params
from dku_model_data_quality.model_utils import check_differences_between_datasets

def process(last_values, ds_test, partition_id):

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
