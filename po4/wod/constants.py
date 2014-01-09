import os.path

from measurements.constants import BASE_DIR

WOD_DIR = os.path.join(BASE_DIR, 'po4/wod')
ANALYSIS_DIR = os.path.join(WOD_DIR, 'analysis')

# CRUISES_PICKLED_FILE = os.path.join(ANALYSIS_DIR, 'cruises_list.ppy')

# MEASUREMENTS_FILE_COORDINATES = os.path.join(ANALYSIS_DIR, 'by_coordinates/measurements_dict_by_coordinates.ppy')
# MEASUREMENTS_FILE_BOXES = os.path.join(ANALYSIS_DIR, 'by_boxes/measurements_dict_by_boxes.ppy')
# 
# ANALYSIS_BY_COORDINATES_DIR = os.path.join(ANALYSIS_DIR, 'by_coordinates/')
# ANALYSIS_BY_COORDINATES_YEAR_DISCARDED_DIR = os.path.join(ANALYSIS_DIR, 'by_coordinates_year_discarded/')

# MEASUREMENT_POINTS_FILE = os.path.join(ANALYSIS_DIR, 'measurement_points.npy')
# MEASUREMENT_RESULTS_FILE = os.path.join(ANALYSIS_DIR, 'measurement_results.npy')
# MEASUREMENT_VARIANCE_FILE = os.path.join(ANALYSIS_DIR, 'measurement_variance.npy')
# MEASUREMENT_STANDARD_DEVIATION_FILE = os.path.join(ANALYSIS_DIR, 'measurement_standard_deviation.npy')
# MEASUREMENT_COVARIANCE_MATRIX_FILE = os.path.join(ANALYSIS_DIR, 'measurement_covariance_matrix.npy')

# VALUES_SEPARATION_VALUES = [1.0/52.0, 360.0/128.0, 180.0/64.0, [0, 50, 120, 220, 360, 550, 790, 1080, 1420, 1810, 2250, 2740, 3280, 3870, 4510, 11000]]

EARTH_RADIUS = 6371 * 10**3