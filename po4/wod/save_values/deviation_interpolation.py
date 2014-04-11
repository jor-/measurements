import os.path
import util.logging

from measurements.po4.wod.deviation.io import save_deviations
from measurements.po4.wod.deviation.constants import MEASUREMENT_DEVIATIONS_INTERPOLATION_FILE


file_prefix = os.path.splitext(MEASUREMENT_DEVIATIONS_INTERPOLATION_FILE)[0]
logging_file = file_prefix + '.log'
deviations_file = file_prefix + '.npy'

with util.logging.Logger(logging_file=logging_file):
    save_deviations(deviations_file=deviations_file)
