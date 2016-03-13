import numpy as np

import measurements.po4.wod.data.results
import measurements.po4.wod.data.values
import measurements.po4.wod.deviation.values
import measurements.util.data
import measurements.util.correlation

import util.cache
import util.logging
logger = util.logging.logger

from measurements.po4.wod.constants import SAMPLE_LSM


## different boxes values

def measurements_same_points(min_values, sample_lsm=SAMPLE_LSM):
    from measurements.po4.wod.correlation.constants import VALUE_DIR, MEASUREMENTS_SAME_POINTS_FILENAME

    def calculate_function():
        m = measurements.po4.wod.data.results.Measurements.load()
        m.categorize_indices_to_lsm(sample_lsm, discard_year=False)
        m.means(return_type='self')
        return m.filter_same_points_except_year(min_values=min_values)

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsSamePoints.load)
    return cache.get_value(MEASUREMENTS_SAME_POINTS_FILENAME.format(min_values=min_values, sample_lsm=sample_lsm), calculate_function)



def sample_values(value_type, min_values, max_year_diff=float('inf'), sample_lsm=SAMPLE_LSM):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))

    from measurements.po4.wod.correlation.constants import VALUE_DIR, DIFFERENT_BOXES_SAMPLE_VALUE_DICT_FILENAME

    def calculate_function():
        ms = measurements_same_points(min_values=min_values, sample_lsm=sample_lsm)
        covariance = ms.correlation_or_covariance(value_type, min_values=min_values, stationary=False, max_year_diff=max_year_diff)
        return covariance

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache.get_value(DIFFERENT_BOXES_SAMPLE_VALUE_DICT_FILENAME.format(type=value_type, max_year_diff=max_year_diff, min_values=min_values, sample_lsm=sample_lsm), calculate_function)



def sample_values_transformed(value_type, min_values, max_year_diff=float('inf'), sample_lsm=SAMPLE_LSM):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))

    from measurements.po4.wod.correlation.constants import VALUE_DIR, DIFFERENT_BOXES_SAMPLE_VALUE_DICT_TRANSFORMED_FILENAME

    def calculate_function():
        value_dict = sample_values(value_type, min_values=min_values, max_year_diff=max_year_diff, sample_lsm=sample_lsm)
        # value_dict = transform_sample_value_dict(value_dict)
        value_dict.coordinates_to_map_indices(sample_lsm, float_indices=False)
        return value_dict

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache.get_value(DIFFERENT_BOXES_SAMPLE_VALUE_DICT_TRANSFORMED_FILENAME.format(type=value_type, max_year_diff=max_year_diff, min_values=min_values, sample_lsm=sample_lsm), calculate_function)


## different boxes covariance dict

def different_boxes_sample_covariances_dict_transformed(min_values, max_year_diff=float('inf'), sample_lsm=SAMPLE_LSM):
    return sample_values_transformed('covariance', min_values, max_year_diff=float('inf'), sample_lsm=SAMPLE_LSM)



## same box covariance 

def same_box_sample_covariances(sample_lsm=SAMPLE_LSM):
    covariances = measurements.po4.wod.deviation.values.concentration_deviation_for_points()**2
    return covariances


