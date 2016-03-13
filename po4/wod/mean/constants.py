import os.path

from ..constants import ANALYSIS_DIR

MEAN_DIR = os.path.join(ANALYSIS_DIR, 'mean')

INTERPOLATED_MEAN_GENERAL_FILENAME = 'interpolated_mean_-_{type}_-_sample_{sample_lsm}_-_scaling_values_{scaling_values}_-_interpolated_{points}_-_min_values_{min_values}_-_{interpolator_setup}.npy'
INTERPOLATED_MEAN_DIRECT_FILENAME = INTERPOLATED_MEAN_GENERAL_FILENAME.format(type='direct', sample_lsm='{sample_lsm}', scaling_values='{scaling_values}', points='{points}', min_values='{min_values}', interpolator_setup='{interpolator_setup}')
INTERPOLATED_MEAN_CONCENTRATION_FILENAME = INTERPOLATED_MEAN_GENERAL_FILENAME.format(type='concentration', sample_lsm='{sample_lsm}', scaling_values='{scaling_values}', points='{points}', min_values='{min_values}', interpolator_setup='{interpolator_setup}')


from measurements.po4.constants import MEAN_MIN_MEASUREMENTS as MIN_MEASUREMENTS
from measurements.po4.wod.constants import SAMPLE_LSM
assert SAMPLE_LSM.t_dim == 12
SCALING_VALUES = (10,1,1,10)