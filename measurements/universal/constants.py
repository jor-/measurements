import math
import os.path
import pathlib

import numpy as np

import matrix.constants
import matrix.sparse.constants

import measurements.land_sea_mask.lsm
from measurements.constants import BASE_DIR

# base dir

MEASUREMENT_DIR = os.path.join(BASE_DIR, '{tracer}', '{data_set}')


# data

DATA_DIR = os.path.join(MEASUREMENT_DIR, 'data')

POINTS_FILE = os.path.join(DATA_DIR, 'measurement_points.npy')
VALUES_FILE = os.path.join(DATA_DIR, 'measurement_values.npy')
MEASUREMENTS_DICT_FILE = os.path.join(DATA_DIR, 'measurements_dict.ppy')

SEPERATOR = '_-_'

NEAR_WATER_DATA_SET_NAME = SEPERATOR.join([
    '{base_data_set_name}',
    '{water_lsm}_water_{max_box_distance_to_water:d}'])

NEAR_WATER_PROJECTION_MASK_FILE = os.path.join(DATA_DIR, 'near_water_projection_mask.npz')
NEAR_WATER_PROJECTION_MATRIX_FILE = os.path.join(DATA_DIR, 'near_water_projection_matrix.{matrix_format}.npz')

INTERPOLATION_FILL_STRATEGY = 'interpolate_{scaling_values}_{interpolator_options}'


# sample lsm

SAMPLE_T_DIM = 12
SAMPLE_LSM = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R(t_dim=SAMPLE_T_DIM)


# mean

MEAN_MIN_MEASUREMENTS = 1

MEAN_DIR = os.path.join(MEASUREMENT_DIR, 'mean')

MEAN_FILE = os.path.join(MEAN_DIR, SEPERATOR.join([
    'concentration_means',
    'for_{target}',
    'sample_{sample_lsm}',
    'min_values_{min_measurements:d}',
    'fill_{fill_strategy}.npy']))

MEAN_ID = SEPERATOR.join([
    'sample_{sample_lsm}',
    'min_values_{min_measurements:d}'])


# quantile

QUANTILE_MIN_MEASUREMENTS = 3

QUANTILE_DIR = os.path.join(MEASUREMENT_DIR, 'quantile')

QUANTILE_FILE = os.path.join(QUANTILE_DIR, SEPERATOR.join([
    '{quantile_type}',
    'for_{target}',
    'sample_{sample_lsm}',
    'min_values_{min_measurements:d}',
    'quantile_{quantile:0<4}',
    'fill_{fill_strategy}.npy']))

QUANTILE_ID = SEPERATOR.join([
    'quantile_{quantile:0<4}',
    'sample_{sample_lsm}',
    'min_values_{min_measurements:d}'])


# deviation

STANDARD_DEVIATION_MIN_MEASUREMENTS = 3

STANDARD_DEVIATION_DIR = os.path.join(MEASUREMENT_DIR, 'standard_deviation')

STANDARD_DEVIATION_FILE = os.path.join(STANDARD_DEVIATION_DIR, SEPERATOR.join([
    '{deviation_type}',
    'for_{target}',
    'sample_{sample_lsm}',
    'min_values_{min_measurements:d}',
    'min_{min_standard_deviation:g}',
    'fill_{fill_strategy}.npy']))

STANDARD_DEVIATION_ID = SEPERATOR.join([
    'sample_{sample_lsm}',
    'min_values_{min_measurements:d}',
    'min_{min_standard_deviation:g}'])


# correlation

CORRELATION_MIN_MEASUREMENTS = 30
CORRELATION_MIN_ABS_VALUE = 0.01
CORRELATION_MAX_ABS_VALUE = 0.99
CORRELATION_DECOMPOSITION_MIN_VALUE_D = 0.1
CORRELATION_DECOMPOSITION_PERMUTATION_METHOD = matrix.sparse.constants.AMD_FILL_REDUCE_PERMUTATION_METHOD
CORRELATION_DECOMPOSITION_TYPE = matrix.constants.LDL_DECOMPOSITION_TYPE
CORRELATION_DTYPE_L = np.float64
# CORRELATION_DECOMPOSITION_MIN_ABS_VALUE_L = 10**math.floor(math.log10(np.finfo(CORRELATION_DTYPE_L).resolution) / 3)
CORRELATION_DECOMPOSITION_MIN_ABS_VALUE_L = 0
CORRELATION_DTYPE = np.dtype(np.float32)
CORRELATION_FORMAT = 'csc'

SAMPLE_CORRELATION_ID = SEPERATOR.join([
    'sample_{sample_lsm}',
    'min_values_{min_measurements_correlation:0>2d}',
    'min_abs_{min_abs_correlation}',
    'max_abs_{max_abs_correlation}',
    'dev:_{standard_deviation_id}'])

DECOMPOSITION_ID_WITHOUT_MIN_ABS_L = SEPERATOR.join([
    '{decomposition_type}',
    '{permutation_method_decomposition_correlation}',
    'min_diag_{min_value_D:.0e}'])

DECOMPOSITION_ID = SEPERATOR.join([
    '{decomposition_type}',
    '{permutation_method_decomposition_correlation}',
    'min_D_{min_value_D:.0e}',
    'min_abs_L_{min_abs_value_L:.0e}'])

CORRELATION_ID_WITHOUT_MIN_ABS_L = SEPERATOR.join([
    'sample_{sample_lsm}',
    'min_values_{min_measurements_correlation:0>2d}',
    'min_abs_{min_abs_correlation}',
    'max_abs_{max_abs_correlation}',
    'decomposition_{decomposition_type}',
    'permutation_{permutation_method_decomposition_correlation}',
    'min_diag_{min_value_D:.0e}',
    'dev:_{standard_deviation_id}'])

CORRELATION_ID = SEPERATOR.join([
    'sample_{sample_lsm}',
    'min_values_{min_measurements_correlation:0>2d}',
    'min_abs_{min_abs_correlation}',
    'max_abs_{max_abs_correlation}',
    '{decomposition_id}',
    'dev:_{standard_deviation_id}'])

# files

CORRELATION_DIR = os.path.join(MEASUREMENT_DIR, 'correlation')

MAP_INDEX_TO_POINT_INDEX_DICT_FILE = os.path.join(
    CORRELATION_DIR,
    'map_indices_to_point_index_dict', SEPERATOR.join([
        'map_indices_to_point_index_dict',
        'sample_{sample_lsm}',
        'year_discarded_{discard_year}.ppy']))

CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILE = os.path.join(
    CORRELATION_DIR,
    'concentrations_same_points_except_year_dict', SEPERATOR.join([
        'concentrations_same_points_except_year_dict',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}.ppy']))

SAMPLE_COVARIANCE_DICT_FILE = os.path.join(
    CORRELATION_DIR,
    'sample_covariance', SEPERATOR.join([
        'sample_covariance_dict.nonstationary',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}.ppy']))

SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'sample_correlation', SEPERATOR.join([
        'sample_correlation.same_box.lower_triangle',
        'sample_{sample_lsm}',
        'min_abs_{min_abs_correlation}',
        'dev:_{standard_deviation_id}',
        '{dtype}.{matrix_format}.npz']))

SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'sample_quantity', SEPERATOR.join([
        'sample_quantity.different_boxes.lower_triangle',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'dev:_{standard_deviation_id}',
        '{dtype}.{matrix_format}.npz']))

SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'sample_correlation', SEPERATOR.join([
        'sample_correlation.different_boxes.lower_triangle',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'dev:_{standard_deviation_id}',
        '{dtype}.{matrix_format}.npz']))

SAMPLE_CORRELATION_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'sample_correlation', SEPERATOR.join([
        'sample_correlation',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'max_abs_{max_abs_correlation}',
        'dev:_{standard_deviation_id}',
        '{dtype}.{matrix_format}.npz']))

CORRELATION_MATRIX_PERMUTATION_VECTOR_FILE = os.path.join(
    CORRELATION_DIR,
    'sample_correlation_permutation_vector', SEPERATOR.join([
        'permutation_vector',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'max_abs_{max_abs_correlation}',
        '{permutation_method_decomposition_correlation}',
        'dev:_{standard_deviation_id}',
        '{dtype}.npy']))

CORRELATION_MATRIX_DECOMPOSITION_FILE = os.path.join(
    CORRELATION_DIR,
    'positive_definite_decomposition', SEPERATOR.join([
        'decomposition',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'max_abs_{max_abs_correlation}',
        '{decomposition_id}',
        'dev:_{standard_deviation_id}',
        '{dtype}.dec']))

CORRELATION_MATRIX_DECOMPOSITION_DELTA_FILE = os.path.join(
    CORRELATION_DIR,
    'positive_definite_decomposition_delta', SEPERATOR.join([
        'delta',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'max_abs_{max_abs_correlation}',
        '{decomposition_id}',
        'dev:_{standard_deviation_id}.npy']))

CORRELATION_MATRIX_DECOMPOSITION_OMEGA_FILE = os.path.join(
    CORRELATION_DIR,
    'positive_definite_decomposition_omega', SEPERATOR.join([
        'omega',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'max_abs_{max_abs_correlation}',
        '{decomposition_id}',
        'dev:_{standard_deviation_id}.npy']))

CORRELATION_MATRIX_POSITIVE_DEFINITE_FILE = os.path.join(
    CORRELATION_DIR,
    'positive_definite', SEPERATOR.join([
        'correlation',
        'sample_{sample_lsm}',
        'min_values_{min_measurements_correlation:0>2d}',
        'min_abs_{min_abs_correlation}',
        'max_abs_{max_abs_correlation}',
        '{decomposition_id}',
        'dev:_{standard_deviation_id}',
        '{dtype}.{matrix_format}.npz']))

# *** correlation arrays *** #

CORRELATION_ARRAY_SAMPLE_CORRELATION_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'array_sample_correlation', SEPERATOR.join([
        'correlation',
        '{sample_correlation_id}',
        '{dtype}',
        'axis_{axis}.npz']))

CORRELATION_ARRAY_CORRELATION_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'array_correlation', SEPERATOR.join([
        'correlation',
        '{correlation_id}',
        '{dtype}',
        'axis_{axis}.npz']))

CORRELATION_LAG_ARRAY_SAMPLE_CORRELATION_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'array_sample_correlation', SEPERATOR.join([
        'correlation_lag',
        '{sample_correlation_id}',
        '{dtype}',
        'axis_{axis}.npz']))

CORRELATION_LAG_ARRAY_CORRELATION_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'array_correlation', SEPERATOR.join([
        'correlation_lag',
        '{correlation_id}',
        '{dtype}',
        'axis_{axis}.npz']))

CORRELATION_LAG_ARRAY_LEXSORT_INDICES_SAMPLE_CORRELATION_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'array_sample_correlation', SEPERATOR.join([
        'correlation_lag_lexsorted_indices',
        '{sample_correlation_id}',
        '{dtype}.npz']))

CORRELATION_LAG_ARRAY_LEXSORT_INDICES_CORRELATION_MATRIX_FILE = os.path.join(
    CORRELATION_DIR,
    'array_correlation', SEPERATOR.join([
        'correlation_lag_lexsorted_indices',
        '{correlation_id}',
        '{dtype}.npz']))

CORRELATION_LAG_INTERQUARTILE_RANGES_SAMPLE_CORRELATION_FILE = os.path.join(
    CORRELATION_DIR,
    'array_sample_correlation', SEPERATOR.join([
        'correlation_lag_interquartile_ranges',
        '{sample_correlation_id}',
        '{dtype}',
        'min_values_{min_values}.npz']))

CORRELATION_LAG_INTERQUARTILE_RANGES_CORRELATION_FILE = os.path.join(
    CORRELATION_DIR,
    'array_correlation', SEPERATOR.join([
        'correlation_lag_interquartile_ranges',
        '{correlation_id}',
        '{dtype}',
        'min_values_{min_values}.npz']))


# *** plots *** #

PLOT_BASE_DIR = str(pathlib.PurePath(BASE_DIR).parent.joinpath('plots').joinpath(pathlib.PurePath(BASE_DIR).name))


def plot_file(file):
    plot_file = str(file).replace(BASE_DIR, PLOT_BASE_DIR, 1)
    plot_file = pathlib.PurePath(plot_file)
    plot_file = plot_file.with_suffix('.svg')
    plot_file = str(plot_file)
    return plot_file
