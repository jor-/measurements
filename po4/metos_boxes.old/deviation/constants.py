import os.path
from measurements.po4.metos_boxes.constants import ANALYSIS_DIR


## constants for sample standard deviation
T_DIM = 12


## constants for interpolation
# AMOUNT_OF_WRAP_AROUND = 0.1
# NUMBER_OF_LINEAR_INTERPOLATOR = 2
# TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0.2
# 
# METOS_BOXES_DEVIATIONS_INTERPOLATED_LIST_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_deviations_interpolated_list_{}_{}_{}_{}.npy'.format('{}', AMOUNT_OF_WRAP_AROUND, NUMBER_OF_LINEAR_INTERPOLATOR, TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR))

# INTERPOLATOR_SETUP = (0.1, 2, 0.1)
INTERPOLATOR_SETUP = (0.1, 2, 0.2)
# METOS_BOXES_DEVIATIONS_INTERPOLATED_LIST_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_deviations_interpolated_list_{}_{}.npy'.format('{}', str(INTERPOLATOR_SETUP).replace(' ','')))
METOS_BOXES_DEVIATIONS_INTERPOLATED_LIST_FILE = os.path.join(ANALYSIS_DIR, 'metos_boxes_deviations_interpolated_list_{}_{}.npy')
