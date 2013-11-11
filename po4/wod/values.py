import numpy as np
import logging
logger = logging.getLogger(__name__)

from .data.measurements import Measurements
from .variance.model import Variance_Model

import util.spherical

from .constants import MEASUREMENT_POINTS_FILE, MEASUREMENT_RESULTS_FILE, MEASUREMENT_VARIANCE_FILE, MEASUREMENT_STANDARD_DEVIATION_FILE


def save_measurements(points_file=MEASUREMENT_POINTS_FILE, results_file=MEASUREMENT_RESULTS_FILE):
    from .constants import MEASUREMENTS_FILE_COORDINATES, VALUES_SEPARATION_VALUES
    
    logger.debug('Loading and calculating measurements.')
    measurements = Measurements()
    measurements.load(MEASUREMENTS_FILE_COORDINATES)
    
    values = measurements.means()
    (n, m) = values.shape
    points = values[:,:m-1]
    results = values[:,m-1]
    
    logger.debug('Saving %d measurement points to %s.' % (n, points_file))
    np.save(points_file, points)
    logger.debug('Saving %d measurement results to %s.' % (n, results_file))
    np.save(results_file, results)
    
    logger.debug('Measurements saved.')


def load_measurement_points(file=MEASUREMENT_POINTS_FILE):
    logger.debug('Loading measurement points from %s.' % file)
    values = np.load(file)
    return values


def load_measurement_results(file=MEASUREMENT_RESULTS_FILE):
    logger.debug('Loading measurement results from %s.' % file)
    values = np.load(file)
    return values



def save_variance(points_file=MEASUREMENT_POINTS_FILE, variance_file=MEASUREMENT_VARIANCE_FILE):
#     from .covariance.constants import VARIANCE_MIN_MEASUREMENTS
    
    points = load_measurement_points(points_file)
#     variance = Variance_Model(minimum_measurements=VARIANCE_MIN_MEASUREMENTS).variance(points)
    variance = Variance_Model().variance(points)
    
    logger.debug('Saving %d variances to %s.' % (len(variance), variance_file))
    np.save(variance_file, variance)
    
    logger.debug('Variances saved.')


def load_variances(file=MEASUREMENT_VARIANCE_FILE):
    logger.debug('Loading variances from %s.' % file)
    values = np.load(file)
    return values



def save_standard_deviation(variance_file=MEASUREMENT_VARIANCE_FILE, standard_deviation_file=MEASUREMENT_STANDARD_DEVIATION_FILE):
    from .covariance.constants import VARIANCE_MIN_MEASUREMENTS
    
    variance = load_variances(variance_file)
    standard_deviation = np.sqrt(variance)
    
    logger.debug('Saving %d standard deviation to %s.' % (len(standard_deviation), standard_deviation_file))
    np.save(standard_deviation_file, standard_deviation)
    
    logger.debug('Standard deviation saved.')


def load_standard_deviations(file=MEASUREMENT_STANDARD_DEVIATION_FILE):
    logger.debug('Loading standard deviations from %s.' % file)
    values = np.load(file)
    return values

    
    