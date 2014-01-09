import numpy as np
import logging
logger = logging.getLogger(__name__)

from .model import Deviation_Model

from ..data.constants import MEASUREMENTS_POINTS_FILE
from .constants import MEASUREMENT_DEVIATIONS_FILE


def save_deviation(measurement_points_file=MEASUREMENTS_POINTS_FILE, deviation_file=MEASUREMENT_DEVIATIONS_FILE):
    
    measurement_points = np.load(measurement_points_file)
    
    logger.debug('Calculating standard deviation for %d points.' % measurement_points.shape[0])
    deviation = Deviation_Model().deviation(measurement_points)
    
    logger.debug('Saving %d standard deviation to %s.' % (len(deviation), deviation_file))
    np.save(deviation_file, deviation)
    
    logger.debug('Standard deviation saved.')


def load_deviations(file=MEASUREMENT_DEVIATIONS_FILE):
    logger.debug('Loading standard deviations from %s.' % file)
    values = np.load(file)
    return values
