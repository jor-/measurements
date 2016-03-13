import os.path

from ..constants import ANALYSIS_DIR

DEVIATION_DIR = os.path.join(ANALYSIS_DIR, 'deviation')

INTERPOLATED_DEVIATION_GENERAL_FILENAME = 'interpolated_deviation_-_{type}_-_sample_{sample_lsm}_-_scaling_values_{scaling_values}_-_interpolated_{points}_-_min_values_{min_values}_-_{interpolator_setup}.npy'
INTERPOLATED_DEVIATION_DIRECT_FILENAME = INTERPOLATED_DEVIATION_GENERAL_FILENAME.format(type='direct', sample_lsm='{sample_lsm}', scaling_values='{scaling_values}', points='{points}', min_values='{min_values}', interpolator_setup='{interpolator_setup}')
INTERPOLATED_DEVIATION_CONCENTRATION_FILENAME = INTERPOLATED_DEVIATION_GENERAL_FILENAME.format(type='concentration', sample_lsm='{sample_lsm}', scaling_values='{scaling_values}', points='{points}', min_values='{min_values}', interpolator_setup='{interpolator_setup}')
INTERPOLATED_DEVIATION_AVERAGE_ERROR_FILENAME = INTERPOLATED_DEVIATION_GENERAL_FILENAME.format(type='average_error', sample_lsm='{sample_lsm}', scaling_values='{scaling_values}', points='{points}', min_values='{min_values}', interpolator_setup='{interpolator_setup}')
INTERPOLATED_DEVIATION_TOTAL_FILENAME = 'interpolated_deviation_-_total_-_sample_{sample_lsm}_-_scaling_values_{scaling_values}_-_interpolated_{points}_-_min_values_{min_values}_-_{concentration_interpolator_setup}_-_{average_error_interpolator_setup}.npy'


from measurements.po4.constants import DEVIATION_MIN_VALUE as MIN_VALUE
from measurements.po4.constants import DEVIATION_MIN_MEASUREMENTS as MIN_MEASUREMENTS
from measurements.po4.wod.constants import SAMPLE_LSM
assert SAMPLE_LSM.t_dim == 12
SCALING_VALUES = (10,1,1,10)

