import numpy as np
import logging
logger = logging.getLogger(__name__)

from .model import Deviation_Model

from .constants import MEASUREMENT_DEVIATIONS_INTERPOLATION_FILE
from ..data.constants import MEASUREMENTS_POINTS_FILE


def save_deviations(measurement_points_file=MEASUREMENTS_POINTS_FILE, deviations_file=MEASUREMENT_DEVIATIONS_INTERPOLATION_FILE):
    
    from .constants import NUMBER_OF_LINEAR_INTERPOLATOR, TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR, AMOUNT_OF_WRAP_AROUND, PARALLEL
    from .constants import SEPARATION_VALUES, T_LEN, X_LEN
    from ..data.constants import MEASUREMENTS_DICT_UNSORTED_FILE
    
    measurement_points = np.load(measurement_points_file)
    
    logger.debug('Calculating standard deviation for %d points.' % measurement_points.shape[0])
    deviation_model = Deviation_Model(measurements_file=MEASUREMENTS_DICT_UNSORTED_FILE, separation_values=SEPARATION_VALUES, t_len=T_LEN, x_len=X_LEN, convert_spherical_to_cartesian=True, wrap_around_amount=AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=NUMBER_OF_LINEAR_INTERPOLATOR, total_overlapping_linear_interpolators=TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR, parallel=PARALLEL)
    deviation = deviation_model.deviation(measurement_points)
    
    logger.debug('Saving %d standard deviation to %s.' % (len(deviation), deviations_file))
    np.save(deviations_file, deviation)
    
    logger.debug('Standard deviation saved.')


def load_deviations(deviations_file=MEASUREMENT_DEVIATIONS_INTERPOLATION_FILE):
    logger.debug('Loading standard deviations from %s.' % deviations_file)
    values = np.load(deviations_file)
    return values
