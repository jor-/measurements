import numpy as np

import measurements.po4.wod.data.results
import measurements.util.data
import measurements.util.correlation

import util.cache
import util.logging
logger = util.logging.logger

from measurements.po4.wod.constants import SAMPLE_LSM


## different boxes values

def measurements_same_points(min_values):
    from measurements.po4.wod.correlation.constants import VALUE_DIR, MEASUREMENTS_SAME_POINTS_FILENAME

    def calculate_function():
        m = measurements.po4.wod.data.results.Measurements.load()
        m.categorize_indices_to_lsm(SAMPLE_LSM, discard_year=False)
        m.means(return_type='self')
        return m.filter_same_points_except_year(min_values=min_values)

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsSamePoints.load)
    return cache.get_value(MEASUREMENTS_SAME_POINTS_FILENAME.format(min_values=min_values), calculate_function)



def sample_values(value_type, min_values, max_year_diff=float('inf')):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))

    from measurements.po4.wod.correlation.constants import VALUE_DIR, SAMPLE_VALUE_DICT_FILENAME

    def calculate_function():
        ms = measurements_same_points(min_values=min_values)
        covariance = ms.correlation_or_covariance(value_type, min_values=min_values, stationary=False, max_year_diff=max_year_diff)
        return covariance

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache.get_value(SAMPLE_VALUE_DICT_FILENAME.format(type=value_type, max_year_diff=max_year_diff, min_values=min_values), calculate_function)



def sample_values_transformed(value_type, min_values, max_year_diff=float('inf')):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))

    from measurements.po4.wod.correlation.constants import VALUE_DIR, SAMPLE_VALUE_DICT_TRANSFORMED_FILENAME

    def calculate_function():
        value_dict = sample_values(value_type, min_values=min_values, max_year_diff=max_year_diff)
        return transform_sample_value_dict(value_dict)

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache.get_value(SAMPLE_VALUE_DICT_TRANSFORMED_FILENAME.format(type=value_type, max_year_diff=max_year_diff, min_values=min_values), calculate_function)


## same box correlation

def same_box_sample_correlations_calculate():
    all_values = measurements.po4.wod.data.results.Measurements.load()
    same_box_sample_correlations_dict = measurements.util.correlation.same_box_sample_correlations(all_values, SAMPLE_LSM)
    return same_box_sample_correlations_dict

def same_box_sample_correlations():
    from measurements.po4.wod.correlation.constants import VALUE_DIR, SAME_BOX_SAMPLE_CORRELATION_DICT_FILENAME
    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.Measurements.load)
    return cache[SAME_BOX_SAMPLE_CORRELATION_DICT_FILENAME, same_box_sample_correlations_calculate]

def same_box_sample_correlations_transformed_calculate():
    value_dict = same_box_sample_correlations()
    return transform_sample_value_dict(value_dict)

def same_box_sample_correlations_transformed():
    from measurements.po4.wod.correlation.constants import VALUE_DIR, SAME_BOX_SAMPLE_CORRELATION_DICT_TRANSFORMED_FILENAME
    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.Measurements.load)
    return cache[SAME_BOX_SAMPLE_CORRELATION_DICT_TRANSFORMED_FILENAME, same_box_sample_correlations_transformed_calculate]
    
    

## different boxes correlation

def different_boxes_sample_correlations_calculate(min_values, max_year_diff=float('inf')):
    ## calculate sample deviations
    values = measurements.po4.wod.data.results.Measurements.load()
    values.categorize_indices_to_lsm(SAMPLE_LSM, discard_year=True)
    sample_deviations = values.deviations(min_values=measurements.constants.DEVIATION_MIN_MEASUREMENTS, return_type='measurements')
    sample_deviations = transform_sample_value_dict(sample_deviations)
    
    def get_deviation(key):
        key = transform_key(key, discard_year=True)
        deviation_list = sample_deviations[key]
        assert len(deviation_list) == 1
        deviation = deviation_list[0]
        return deviation
    
    ## calculate different boxes correlations
    sample_covariances = sample_values('covariance', min_values, max_year_diff=max_year_diff)
    sample_correlation = measurements.util.data.MeasurementsCovariance()
    for key in sample_covariances.keys():
        ## get number_of_measurements and covariance
        covariance_list = sample_covariances[key]
        assert len(covariance_list) == 1
        assert len(covariance_list[0]) == 2
        number_of_measurements, covariance = covariance_list[0]
        ## get deviations
        assert len(key) == 2
        deviations = np.array([get_deviation(key[0]), get_deviation(key[1])])
        ## calculate and insert number_of_measurements and correlation
        if np.all(deviations > 0):
            correlation = covariance / deviations.prod()
            # assert correlation >= -1 and correlation <= 1
            sample_correlation.append_value(key, [number_of_measurements, correlation])
    
    return sample_correlation

def different_boxes_sample_correlations(min_values, max_year_diff=float('inf')):
    from measurements.po4.wod.correlation.constants import VALUE_DIR, DIFFERENT_BOXES_SAMPLE_CORRELATION_DICT_FILENAME
    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache[DIFFERENT_BOXES_SAMPLE_CORRELATION_DICT_FILENAME.format(min_values=min_values, max_year_diff=max_year_diff), lambda: different_boxes_sample_correlations_calculate(min_values=min_values, max_year_diff=max_year_diff)]

def different_boxes_sample_correlations_transformed_calculate(min_values, max_year_diff=float('inf')):
    value_dict = different_boxes_sample_correlations(min_values=min_values, max_year_diff=max_year_diff)
    return transform_sample_value_dict(value_dict)

def different_boxes_sample_correlations_transformed(min_values, max_year_diff=float('inf')):
    from measurements.po4.wod.correlation.constants import VALUE_DIR, DIFFERENT_BOXES_SAMPLE_CORRELATION_DICT_TRANSFORMED_FILENAME
    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache[DIFFERENT_BOXES_SAMPLE_CORRELATION_DICT_TRANSFORMED_FILENAME.format(min_values=min_values, max_year_diff=max_year_diff), lambda: different_boxes_sample_correlations_transformed_calculate(min_values=min_values, max_year_diff=max_year_diff)]

    

## util

def transform_sample_value_dict(value_dict):
    value_dict.coordinates_to_map_indices(SAMPLE_LSM)
    value_dict.keys_to_int_keys(np.int32)
    return value_dict


def transform_key(key, discard_year=False):
    key = SAMPLE_LSM.coordinate_to_map_index(*key, discard_year=discard_year)
    key = tuple(np.array(np.round(key), dtype=np.int32))
    return key

