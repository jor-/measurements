import measurements.util.data


def sample_values_transformed(value_type, min_values, max_year_diff):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))
    return measurements.util.data.MeasurementsCovariance()


def different_boxes_sample_correlations_transformed(min_values, max_year_diff=float('inf')):
    return measurements.util.data.MeasurementsCovariance()


def same_box_sample_correlations_transformed():
    return measurements.util.data.Measurements()

