import numpy as np

import logging
logger = logging.getLogger(__name__)

from measurements.po4.wod.data.results import Measurements_Unsorted

from ..data.constants import MEASUREMENTS_DICT_UNSORTED_FILE
from .constants import SEPARATION_VALUES, MEASUREMENT_DEVIATIONS_ESTIMATION_FILE
from measurements.po4.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE



def deviations_from_measurements(separation_values=SEPARATION_VALUES, minimum_measurements=DEVIATION_MIN_MEASUREMENTS, min_standard_deviation=DEVIATION_MIN_VALUE, measurements_file=MEASUREMENTS_DICT_UNSORTED_FILE):
    
    logger.debug('Calculating deviation for measurements from {} with separation value {}, min measurements {} and min standard deviation {}.'.format(measurements_file, separation_values, minimum_measurements, min_standard_deviation))
    
    ## calculate sample standard deviation
    m = Measurements_Unsorted()
    m.load(measurements_file)
    m.discard_year()
    m.categorize_indices(separation_values)
    deviations = m.deviations(minimum_measurements=minimum_measurements)
    logging.debug('Got standard deviation estimation at {} points.'.format(deviations.shape[0]))
    
    ## apply lower bound
    lower_bound_mask = deviations[:, 4] < min_standard_deviation
    deviations[lower_bound_mask, 4] = min_standard_deviation
    logging.debug('{} standard deviation estimation are below {}, so set to this bound.'.format((lower_bound_mask).sum(), min_standard_deviation))
    
    return deviations


def save_deviations_from_measurements(deviations_file=MEASUREMENT_DEVIATIONS_ESTIMATION_FILE, separation_values=SEPARATION_VALUES, minimum_measurements=DEVIATION_MIN_MEASUREMENTS, min_standard_deviation=DEVIATION_MIN_VALUE, measurements_file=MEASUREMENTS_DICT_UNSORTED_FILE):
    
    deviations = deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, min_standard_deviation=min_standard_deviation, measurements_file=measurements_file)
    np.save(deviations_file, deviations)
    
    logger.debug('Deviation for measurements from {} box dict saved at {} with separation value {} and min measurements {}.'.format(measurements_file, deviations_file, separation_values, minimum_measurements))


def load_deviations_from_measurements(deviations_file=MEASUREMENT_DEVIATIONS_ESTIMATION_FILE):
    deviations = np.load(deviations_file)
    logger.debug('Deviation for measurements loaded from {}.'.format(deviations_file))
    return deviations
