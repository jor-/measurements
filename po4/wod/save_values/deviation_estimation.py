import os.path
import util.logging

from measurements.po4.wod.deviation.estimation import save_deviations_from_measurements
from measurements.po4.wod.deviation.constants import MEASUREMENT_DEVIATIONS_ESTIMATION_FILE


file_prefix = os.path.splitext(MEASUREMENT_DEVIATIONS_ESTIMATION_FILE)[0]
logging_file = file_prefix + '.log'
deviations_file = file_prefix + '.npy'

with util.logging.Logger(logging_file=logging_file):
    save_deviations_from_measurements(deviations_file=deviations_file)