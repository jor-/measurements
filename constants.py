import os.path

import ndop.constants
BASE_DIR = os.path.join(ndop.constants.BASE_DIR, 'measurements')

## earth
EARTH_RADIUS = 6371 * 10**3
MAX_SEA_DEPTH = 11 * 10**3

## mean
MEAN_MIN_MEASUREMENTS = 3

## deviation
DEVIATION_MIN_MEASUREMENTS = 5
DEVIATION_MIN_VALUE = 0.05

## correlation
# CORRELATION_QUANTITY_SAME_BOX = 99
# CORRELATION_SAME_BOX = 0.95
CORRELATION_MIN_ABS_VALUE = 0.01
CORRELATION_MAX_ABS_VALUE = 0.95
CORRELATION_MIN_DIAG_VALUE_POSITIVE_DEFINITE_APPROXIMATION = 0.01

## cache filenames
MEASUREMENTS_POINTS_FILENAME = 'measurement_lexsorted_points.npy'
MEASUREMENTS_RESULTS_FILENAME = 'measurement_lexsorted_results.npy'
MEASUREMENTS_POINTS_ARE_NEAR_WATER_FILENAME = 'measurement_lexsorted_points_near_water_mask_-_{lsm}_-_max_land_boxes_{max_land_boxes}.npy'