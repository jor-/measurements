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


## mean

MEAN_MIN_MEASUREMENTS = 2

MEAN_DIR = os.path.join(MEASUREMENT_DIR, 'mean')


## deviation

DEVIATION_MIN_MEASUREMENTS = 3

DEVIATION_DIR = os.path.join(MEASUREMENT_DIR, 'deviation')


# fill averaged files

MEAN_FILL_AVERAGED_FILENAME = 'concentration_mean_-_fill_average_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}.npy'
MEAN_FILL_AVERAGED_FILE = os.path.join(MEAN_DIR, MEAN_FILL_AVERAGED_FILENAME)


CONCENTRATION_DEVIATION_FILL_AVERAGED_FILENAME = 'concentration_deviation_-_fill_average_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_min_deviation_{min_deviation:g}.npy'
CONCENTRATION_DEVIATION_FILL_AVERAGED_FILE = os.path.join(DEVIATION_DIR, CONCENTRATION_DEVIATION_FILL_AVERAGED_FILENAME)

NOISE_DEVIATION_FILL_AVERAGED_FILENAME = 'noise_deviation_-_fill_average_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_min_deviation_{min_deviation:g}.npy'
NOISE_DEVIATION_FILL_AVERAGED_FILE = os.path.join(DEVIATION_DIR, NOISE_DEVIATION_FILL_AVERAGED_FILENAME)

TOTAL_DEVIATION_FILL_AVERAGED_FILENAME = 'total_deviation_-_fill_average_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_min_deviation_{min_deviation:g}.npy'
TOTAL_DEVIATION_FILL_AVERAGED_FILE = os.path.join(DEVIATION_DIR, TOTAL_DEVIATION_FILL_AVERAGED_FILENAME)


# fill interpolated files

MEAN_FILL_INTERPOLATED_FILENAME = 'concentration_mean_-_fill_interpolation_for_{interpolation_target}_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_scaling_values_{scaling_values}_-_interpolator_setup_{interpolator_setup}.npy'
MEAN_FILL_INTERPOLATED_FILE = os.path.join(MEAN_DIR, MEAN_FILL_INTERPOLATED_FILENAME)


CONCENTRATION_DEVIATION_FILL_INTERPOLATED_FILENAME = 'concentration_deviation_-_fill_interpolation_for_{interpolation_target}_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_min_deviation_{min_deviation:g}_-_scaling_values_{scaling_values}_-_interpolator_setup_{concentration_interpolator_setup}.npy'
CONCENTRATION_DEVIATION_FILL_INTERPOLATED_FILE = os.path.join(DEVIATION_DIR, CONCENTRATION_DEVIATION_FILL_INTERPOLATED_FILENAME)

NOISE_DEVIATION_FILL_INTERPOLATED_FILENAME = 'noise_deviation_-_fill_interpolation_for_{interpolation_target}_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_min_deviation_{min_deviation:g}_-_scaling_values_{scaling_values}_-_interpolator_setup_{average_noise_interpolator_setup}.npy'
NOISE_DEVIATION_FILL_INTERPOLATED_FILE = os.path.join(DEVIATION_DIR, NOISE_DEVIATION_FILL_INTERPOLATED_FILENAME)

TOTAL_DEVIATION_FILL_INTERPOLATED_FILENAME = 'total_deviation_-_fill_interpolation_for_{interpolation_target}_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:d}_-_min_deviation_{min_deviation:g}_-_scaling_values_{scaling_values}_-_interpolator_setup_{concentration_interpolator_setup}_{average_noise_interpolator_setup}.npy'
TOTAL_DEVIATION_FILL_INTERPOLATED_FILE = os.path.join(DEVIATION_DIR, TOTAL_DEVIATION_FILL_INTERPOLATED_FILENAME)


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

CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILENAME = 'concentrations_same_points_except_year_dict_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}.ppy'
CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILE = os.path.join(CORRELATION_DIR, CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILENAME)

SAMPLE_CORRELATION_OR_COVARIANCE_DICT_FILENAME = 'sample_{value_type}_dict.nonstationary_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}_-_max_year_diff_{max_year_diff:0>2}.ppy'
SAMPLE_CORRELATION_OR_COVARIANCE_DICT_FILE = os.path.join(CORRELATION_DIR, SAMPLE_CORRELATION_OR_COVARIANCE_DICT_FILENAME)

SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME = 'sample_quantity_matrix.different_boxes.lower_triangle_matrix_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}_-_max_year_diff_{max_year_diff:0>2}_-_min_abs_correlation_{min_abs_correlation}_-_{dtype}.{matrix_format}.npz'
SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME)

SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME = 'sample_correlation_matrix.different_boxes.lower_triangle_matrix_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}_-_max_year_diff_{max_year_diff:0>2}_-_min_abs_correlation_{min_abs_correlation}_-_{dtype}.{matrix_format}.npz'
SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILENAME)

SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILENAME = 'sample_correlation_matrix.same_box.lower_triangle_matrix_-_sample_{sample_lsm}_-_min_abs_correlation_{min_abs_correlation}_-_{dtype}.{matrix_format}.npz'
SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILENAME)

SAMPLE_CORRELATION_MATRIX_FILENAME = 'sample_correlation_matrix_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}_-_min_abs_correlation_{min_abs_correlation}_-_max_abs_correlation_{max_abs_correlation}_-_{dtype}.{matrix_format}.npz'
SAMPLE_CORRELATION_MATRIX_FILE = os.path.join(CORRELATION_DIR, SAMPLE_CORRELATION_MATRIX_FILENAME)

CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME = 'correlation_matrix.positive_definite_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}_-_min_abs_correlation_{min_abs_correlation}_-_max_abs_correlation_{max_abs_correlation}_-_{ordering_method}_ordering.reordering_{reordering}_-_min_diag_{cholesky_min_diag_value:.0e}_-_{dtype}.{matrix_format}.npz'
CORRELATION_MATRIX_POSITIVE_DEFINITE_FILE = os.path.join(CORRELATION_DIR, CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME)

CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME = 'correlation_matrix.positive_definite.reduction_factors_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}_-_min_abs_correlation_{min_abs_correlation}_-_max_abs_correlation_{max_abs_correlation}_-_{ordering_method}_ordering.reordering_{reordering}_-_min_diag_{cholesky_min_diag_value:.0e}_-_{dtype}.{matrix_format}.npy'
CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILE = os.path.join(CORRELATION_DIR, CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME)

CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME = 'correlation_matrix.positive_definite.cholesky_factor_{factor_type}_-_sample_{sample_lsm}_-_min_measurements_{min_measurements:0>2d}_-_min_abs_correlation_{min_abs_correlation}_-_max_abs_correlation_{max_abs_correlation}_-_{ordering_method}_ordering.reordering_{reordering}_-_min_diag_{cholesky_min_diag_value:.0e}_-_{dtype}.{matrix_format}.npz'
CORRELATION_MATRIX_CHOLESKY_FACTOR_FILE = os.path.join(CORRELATION_DIR, CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME)

