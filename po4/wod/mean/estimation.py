import numpy as np

import logging
logger = logging.getLogger(__name__)

from measurements.po4.wod.data.results import Measurements

from ..data.constants import MEASUREMENTS_FILE
from .constants import SEPARATION_VALUES, MEASUREMENT_MEANS_ESTIMATION_FILE, MIN_MEASUREMENTS



def means_from_measurements(separation_values=SEPARATION_VALUES, minimum_measurements=MIN_MEASUREMENTS, measurements_file=MEASUREMENTS_FILE):
    from .constants import T_RANGE, X_RANGE
    
    m = Measurements()
    m.load(measurements_file)
    m.discard_year()
    m.categorize_indices(separation_values, t_wrap_around_range=T_RANGE, x_wrap_around_range=X_RANGE)
    means = m.means(minimum_measurements=minimum_measurements)
    
    return means

def save_means_from_measurements(means_file=MEASUREMENT_MEANS_ESTIMATION_FILE, separation_values=SEPARATION_VALUES, minimum_measurements=MIN_MEASUREMENTS, measurements_file=MEASUREMENTS_FILE):
    means = means_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file)
    np.save(means_file, means)
