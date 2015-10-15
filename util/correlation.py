import numpy as np

import measurements.util.data
import measurements.constants

import util.logging
logger = util.logging.logger


def same_box_sample_correlations(all_values, lsm):
    ## calculate all and seasonal variances
    all_values.categorize_indices_to_lsm(lsm, discard_year=False)
    seasonal_values = all_values.means(min_values=measurements.constants.MEAN_MIN_MEASUREMENTS, return_type='measurements')
    seasonal_values.discard_year()
    seasonal_variances = seasonal_values.variances(min_values=measurements.constants.DEVIATION_MIN_MEASUREMENTS, return_type='measurements')
    all_values.discard_year()
    all_variances = all_values.variances(min_values=measurements.constants.DEVIATION_MIN_MEASUREMENTS, return_type='measurements')
    
    ## calculate same box correlations
    same_box_correlations = measurements.util.data.Measurements()
    for key in seasonal_variances.keys():
        seasonal_list = seasonal_variances[key]
        assert len(seasonal_list) == 1
        seasonal_variance = seasonal_list[0]
        all_list = all_variances[key]
        assert len(all_list) == 1
        all_variance = all_list[0]
        if all_variance > 0:
            correlation = seasonal_variance / all_variance
            correlation = min(correlation, measurements.constants.CORRELATION_MAX_ABS_VALUE)
            same_box_correlations.append_value(key, correlation)
    
    return same_box_correlations
