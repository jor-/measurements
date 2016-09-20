import os.path

import numpy as np

from measurements.constants import BASE_DIR

## base dir

MEASUREMENT_DIR = os.path.join(BASE_DIR, '{tracer}', '{data_set}')


## data

DATA_DIR = os.path.join(MEASUREMENT_DIR, 'data')

POINTS_FILE = os.path.join(DATA_DIR, 'measurement_points.npy')
VALUES_FILE = os.path.join(DATA_DIR, 'measurement_values.npy')
MEASUREMENTS_DICT_FILE = os.path.join(DATA_DIR, 'measurements_dict.ppy')

NEAR_WATER_DATA_SET_NAME = '{base_data_set_name}_-_{water_lsm}_water_{max_box_distance_to_water:d}'
NEAR_WATER_PROJECTION_MASK_FILE = os.path.join(DATA_DIR, 'near_water_projection_matrix.{matrix_format}.npz')

INTERPOLATION_FILL_STRATEGY = 'interpolate_{scaling_values}_{interpolator_options}'


## mean

MEAN_MIN_MEASUREMENTS = 2

MEAN_DIR = os.path.join(MEASUREMENT_DIR, 'mean')

MEAN_FILENAME = 'concentration_mean_-_for_{target}_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_fill_{fill_strategy}.npy'
MEAN_FILE = os.path.join(MEAN_DIR, MEAN_FILENAME)


## deviation

DEVIATION_MIN_MEASUREMENTS = 3

DEVIATION_DIR = os.path.join(MEASUREMENT_DIR, 'deviation')

DEVIATION_FILENAME = '{deviation_type}_-_for_{target}_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_min_deviation_{min_standard_deviation:g}_-_fill_{fill_strategy}.npy'
DEVIATION_FILE = os.path.join(DEVIATION_DIR, DEVIATION_FILENAME)


## correlation

CORRELATION_MIN_MEASUREMENTS = 30
CORRELATION_MIN_ABS_VALUE = 0.01
CORRELATION_MAX_ABS_VALUE = 0.99
CORRELATION_CHOLESKY_MIN_DIAG_VALUE = 0.1
CORRELATION_CHOLESKY_ORDERING_METHOD = 'default'
CORRELATION_CHOLEKSY_REORDER_AFTER_EACH_STEP = True
CORRELATION_DTYPE = np.dtype(np.float32)
CORRELATION_FORMAT = 'csc'

# files

CORRELATION_DIR = os.path.join(MEASUREMENT_DIR, 'correlation')


MAP_INDEX_TO_POINT_INDEX_DICT_FILENAME = 'map_indices_to_point_index_dict_-_sample_{sample_lsm}_-_year_discarded_{discard_year}.ppy'
MAP_INDEX_TO_POINT_INDEX_DICT_FILE = os.path.join(CORRELATION_DIR, MAP_INDEX_TO_POINT_INDEX_DICT_FILENAME)

CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILENAME = 'concentrations_same_points_except_year_dict_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}.ppy'
CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILE = os.path.join(CORRELATION_DIR, CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILENAME)

SAMPLE_COVARIANCE_DICT_FILENAME = 'sample_covariance_dict.nonstationary_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}_-_max_year_diff_{max_year_diff:0>2}.ppy'
SAMPLE_COVARIANCE_DICT_FILE = os.path.join(CORRELATION_DIR, SAMPLE_COVARIANCE_DICT_FILENAME)

DEVIATION_DESCRIPTION = 'min_values_{min_measurements:d}_-_min_{min_standard_deviation:g}_-_fill_{fill_strategy}'

SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILENAME = 'sample_correlation.same_box.lower_triangle_-_sample_{sample_lsm}_-_min_abs_{min_abs_correlation}_-_deviation_{standard_deviation_description}_-_{dtype}.{matrix_format}.npz'
SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILENAME)

SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME = 'sample_quantity.different_boxes.lower_triangle_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}_-_max_year_diff_{max_year_diff:0>2}_-_min_abs_{min_abs_correlation}_-_deviation_{standard_deviation_description}_-_{dtype}.{matrix_format}.npz'
SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME)

SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME = 'sample_correlation.different_boxes.lower_triangle_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}_-_max_year_diff_{max_year_diff:0>2}_-_min_abs_{min_abs_correlation}_-_deviation_{standard_deviation_description}_-_{dtype}.{matrix_format}.npz'
SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME)

SAMPLE_CORRELATION_MATRIX_FILENAME = 'sample_correlation_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}_-_min_abs_{min_abs_correlation}_-_max_abs_{max_abs_correlation}_-_deviation_{standard_deviation_description}_-_{dtype}.{matrix_format}.npz'
SAMPLE_CORRELATION_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_CORRELATION_MATRIX_FILENAME)

CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME = 'correlation.positive_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}_-_min_abs_{min_abs_correlation}_-_max_abs_{max_abs_correlation}_-_{ordering_method}_ordering.reordering_{reordering}_-_min_diag_{cholesky_min_diag_value:.0e}_-_deviation_{standard_deviation_description}_-_{dtype}.{matrix_format}.npz'
CORRELATION_MATRIX_POSITIVE_DEFINITE_FILE = os.path.join(CORRELATION_DIR, CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME)

CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME = 'correlation.positive.reduction_factors_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}_-_min_abs_{min_abs_correlation}_-_max_abs_{max_abs_correlation}_-_{ordering_method}_ordering.reordering_{reordering}_-_min_diag_{cholesky_min_diag_value:.0e}_-_min_measurements_deviation_{min_measurements_standard_deviation:d}_-_min_deviation_{min_standard_deviation:g}_-_deviation_{standard_deviation_description}.npy'
CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILE = os.path.join(CORRELATION_DIR, CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME)

CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME = 'correlation.positive.cholesky_{factor_type}_-_sample_{sample_lsm}_-_min_values_{min_measurements_correlation:0>2d}_-_min_abs_{min_abs_correlation}_-_max_abs_{max_abs_correlation}_-_{ordering_method}_ordering.reordering_{reordering}_-_min_diag_{cholesky_min_diag_value:.0e}_-_deviation_{standard_deviation_description}_-_{dtype}.{matrix_format}.npz'
CORRELATION_MATRIX_CHOLESKY_FACTOR_FILE = os.path.join(CORRELATION_DIR, CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME)

