import os.path
from measurements.all.pw.constants import BASE_DIR

T_DIM = 12

from measurements.po4.metos_boxes.deviation.constants import INTERPOLATOR_SETUP
# 
# METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE = os.path.join(BASE_DIR, 'metos_boxes_deviations_interpolated_map_{}_{}.npy'.format('{}', str(INTERPOLATOR_SETUP).replace(' ','')))
METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE = os.path.join(BASE_DIR, 'metos_boxes_deviations_interpolated_map_{}_{}.npy')