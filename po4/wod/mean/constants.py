import os.path

from ..constants import ANALYSIS_DIR
from ..variance.constants import MIN_MEASUREMENTS, SEPARATION_VALUES, X_RANGE, T_RANGE


MEASUREMENT_MEANS_ESTIMATION_FILE = os.path.join(ANALYSIS_DIR, 'measurement_means_estimation.npy')