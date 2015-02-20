import os.path
from measurements.po4.metos_boxes.constants import ANALYSIS_DIR


## constants for interpolation
# AMOUNT_OF_WRAP_AROUND = 0.1
# # NUMBER_OF_LINEAR_INTERPOLATOR = 2
# # TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0.2
# # NUMBER_OF_LINEAR_INTERPOLATOR = 2
# # TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0.1
# NUMBER_OF_LINEAR_INTERPOLATORS = 0
# TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATORS = 0
# 
# # METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_means_interpolated_list_{}_{}_{}_{}.npy'.format('{}', AMOUNT_OF_WRAP_AROUND, NUMBER_OF_LINEAR_INTERPOLATOR, TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR))
# 
# METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_means_interpolated_list_{}_{}_{}_{}.npy')
# 
# METOS_BOXES_MEANS_INTERPOLATED_MAP_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_means_interpolated_map_{}_{}_{}_{}.npy')


INTERPOLATOR_SETUP = (0.1, 0, 0)

# METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_means_interpolated_list_{}_{}.npy')
# METOS_BOXES_MEANS_INTERPOLATED_MAP_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_means_interpolated_map_{}_{}.npy')
METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_means_{min_measurements}_interpolated_list_{time_dim}_{interpolator_setup}.npy')
METOS_BOXES_MEANS_INTERPOLATED_MAP_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_means_{min_measurements}_interpolated_map_{time_dim}_{interpolator_setup}.npy')