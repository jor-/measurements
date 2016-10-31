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

SEPERATOR = '_-_'

NEAR_WATER_DATA_SET_NAME = SEPERATOR.join([
'{base_data_set_name}',
'{water_lsm}_water_{max_box_distance_to_water:d}'])

NEAR_WATER_PROJECTION_MASK_FILE = os.path.join(DATA_DIR, 'near_water_projection_matrix.{matrix_format}.npz')

INTERPOLATION_FILL_STRATEGY = 'interpolate_{scaling_values}_{interpolator_options}'


## mean

MEAN_MIN_MEASUREMENTS = 2

MEAN_DIR = os.path.join(MEASUREMENT_DIR, 'mean')

MEAN_FILE = os.path.join(MEAN_DIR, SEPERATOR.join([
'concentration_mean',
'for_{target}',
'sample_{sample_lsm}',
'min_values_{min_measurements:d}',
'fill_{fill_strategy}.npy']))

MEAN_ID = SEPERATOR.join([
'sample_{sample_lsm}',
'min_values_{min_measurements:d}'])


## deviation

DEVIATION_MIN_MEASUREMENTS = 3

DEVIATION_DIR = os.path.join(MEASUREMENT_DIR, 'deviation')

DEVIATION_FILE = os.path.join(DEVIATION_DIR, SEPERATOR.join([
'{deviation_type}',
'for_{target}',
'sample_{sample_lsm}',
'min_values_{min_measurements:d}',
'min_{min_standard_deviation:g}',
'fill_{fill_strategy}.npy']))

DEVIATION_ID = SEPERATOR.join([
'sample_{sample_lsm}',
'min_values_{min_measurements:d}',
'min_{min_standard_deviation:g}'])


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

CORRELATION_ID = SEPERATOR.join([
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'min_abs_{min_abs_correlation}',
'max_abs_{max_abs_correlation}',
'ordering_{cholesky_ordering_method_correlation}_{cholesky_reordering_correlation:d}',
'min_diag_{cholesky_min_diag_value:.0e}',
'dev:_{standard_deviation_id}'])

CORRELATION_DIR = os.path.join(MEASUREMENT_DIR, 'correlation')

MAP_INDEX_TO_POINT_INDEX_DICT_FILE = os.path.join(CORRELATION_DIR,
'map_indices_to_point_index_dict', SEPERATOR.join([
'map_indices_to_point_index_dict',
'sample_{sample_lsm}',
'year_discarded_{discard_year}.ppy']))

CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILE = os.path.join(CORRELATION_DIR,
'concentrations_same_points_except_year_dict', SEPERATOR.join([
'concentrations_same_points_except_year_dict', 
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}.ppy']))

SAMPLE_COVARIANCE_DICT_FILE = os.path.join(CORRELATION_DIR,
'sample_covariance', SEPERATOR.join([
'sample_covariance_dict.nonstationary',
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'max_year_diff_{max_year_diff:0>2}.ppy']))

SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR,
'sample_correlation', SEPERATOR.join([
'sample_correlation.same_box.lower_triangle',
'sample_{sample_lsm}',
'min_abs_{min_abs_correlation}',
'dev:_{standard_deviation_id}',
'{dtype}.{matrix_format}.npz']))

SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR,
'sample_quantity', SEPERATOR.join([
'sample_quantity.different_boxes.lower_triangle',
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'max_year_diff_{max_year_diff:0>2}',
'min_abs_{min_abs_correlation}',
'dev:_{standard_deviation_id}',
'{dtype}.{matrix_format}.npz']))

SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR,
'sample_correlation', SEPERATOR.join([
'sample_correlation.different_boxes.lower_triangle',
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'max_year_diff_{max_year_diff:0>2}',
'min_abs_{min_abs_correlation}',
'dev:_{standard_deviation_id}',
'{dtype}.{matrix_format}.npz']))

SAMPLE_CORRELATION_MATRIX_FILE = os.path.join(CORRELATION_DIR,
'sample_correlation', SEPERATOR.join([
'sample_correlation',
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'min_abs_{min_abs_correlation}',
'max_abs_{max_abs_correlation}',
'dev:_{standard_deviation_id}',
'{dtype}.{matrix_format}.npz']))

CORRELATION_MATRIX_POSITIVE_DEFINITE_FILE = os.path.join(CORRELATION_DIR,
'positive_definite', SEPERATOR.join([
'correlation',
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'min_abs_{min_abs_correlation}',
'max_abs_{max_abs_correlation}',
'ordering_{cholesky_ordering_method_correlation}_{cholesky_reordering_correlation:d}',
'min_diag_{cholesky_min_diag_value:.0e}',
'dev:_{standard_deviation_id}',
'{dtype}.{matrix_format}.npz']))

CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILE = os.path.join(CORRELATION_DIR,
'positive_definite', SEPERATOR.join([
'reduction_factors',
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'min_abs_{min_abs_correlation}',
'max_abs_{max_abs_correlation}',
'ordering_{cholesky_ordering_method_correlation}_{cholesky_reordering_correlation:d}',
'min_diag_{cholesky_min_diag_value:.0e}',
'dev:_{standard_deviation_id}.npy']))

CORRELATION_MATRIX_CHOLESKY_FACTOR_FILE = os.path.join(CORRELATION_DIR,
'positive_definite', SEPERATOR.join([
'cholesky_{factor_type}',
'sample_{sample_lsm}',
'min_values_{min_measurements_correlation:0>2d}',
'min_abs_{min_abs_correlation}',
'max_abs_{max_abs_correlation}',
'ordering_{cholesky_ordering_method_correlation}_{cholesky_reordering_correlation:d}',
'min_diag_{cholesky_min_diag_value:.0e}',
'dev:_{standard_deviation_id}',
'{dtype}.{matrix_format}.npz']))
