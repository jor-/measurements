import os.path
import util.logging

from measurements.po4.metos_boxes.data.io import save_measurement_boxes_dict as save
from measurements.po4.metos_boxes.data.constants import METOS_BOXES_DICT_FILE as file


file_prefix = os.path.splitext(file)[0]
log_file = file_prefix + '.log'

with util.logging.Logger(log_file=log_file, disp_stdout=False):
    save()