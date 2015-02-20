import os.path
import util.logging

from measurements.all.metos_boxes.deviation.io import save_interpolated_deviation_map as save
from measurements.all.metos_boxes.deviation.constants import METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE, INTERPOLATOR_SETUP
T_DIM = 52
file = METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE.format(T_DIM, str(INTERPOLATOR_SETUP).replace(' ',''))

file_prefix = os.path.splitext(file)[0]
log_file = file_prefix + '.log'

with util.logging.Logger(log_file=log_file, disp_stdout=False):
    save(t_dim=T_DIM)
# with util.logging.Logger(log_file=log_file.format(12), disp_stdout=False):
#     save(t_dim=12)
# with util.logging.Logger(log_file=log_file.format(52), disp_stdout=False):
#     save(t_dim=52)
