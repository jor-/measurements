import numpy as np

import logging
logger = logging.getLogger(__name__)

from measurements.po4.wod.data.results import Measurements

from ..data.constants import MEASUREMENTS_DICT_FILE
from .constants import SEPARATION_VALUES, MEASUREMENT_DEVIATIONS_ESTIMATION_FILE, MIN_MEASUREMENTS, T_RANGE, X_RANGE



def deviations_from_measurements(separation_values=SEPARATION_VALUES, minimum_measurements=MIN_MEASUREMENTS, measurements_file=MEASUREMENTS_DICT_FILE, t_range=T_RANGE, x_range=X_RANGE):
    
    logger.debug('Calculationg deviation for measurements from {} with separation value {} and min_measurements {}'.format(measurements_file, separation_values, minimum_measurements))
    
    m = Measurements()
    m.load(measurements_file)
    m.discard_year()
    m.categorize_indices(separation_values, wrap_around_ranges=(t_range, x_range))
    deviations = m.deviations(minimum_measurements=minimum_measurements)
    
    # discard negative values
    deviations = deviations[deviations[:, 4] > 0]
    
    return deviations


def save_deviations_from_measurements(deviations_file=MEASUREMENT_DEVIATIONS_ESTIMATION_FILE, separation_values=SEPARATION_VALUES, minimum_measurements=MIN_MEASUREMENTS, measurements_file=MEASUREMENTS_DICT_FILE):
    
    deviations = deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file)
    np.save(deviations_file, deviations)
    
    logger.debug('Deviation for measurements from {} box dict saved at {} with separation value {} and min measurments {}'.format(measurements_file, deviations_file, separation_values, minimum_measurements))


def load_deviations_from_measurements(deviations_file=MEASUREMENT_DEVIATIONS_ESTIMATION_FILE):
    deviations = np.load(deviations_file)
    logger.debug('Deviation for measurements loaded from {}.'.format(deviations_file))
    return deviations



# def space_deviation_from_measurements(separation_values, minimum_measurements=10):
#     from ..constants import MEASUREMENTS_FILE_COORDINATES
#     from .constants import SEPARATION_VALUES, X_RANGE
#     
#     m = Measurements()
#     m.load(MEASUREMENTS_FILE_COORDINATES)
#     m.discard_time()
#     if separation_values is None:
#         separation_values = SEPARATION_VALUES
#     m.categorize_indices(separation_values, X_RANGE)
#     deviation = m.deviations(minimum_measurements=minimum_measurements)
#     
#     # discard negative values
#     deviation = deviation[deviation[:, 4] > 0]
#     # discard time
#     deviation = deviation[:, 1:5]
#     
#     return deviation


# def space_deviation_from_measurements_by_boxes(minimum_measurements=10, interpolate=True):
#     from ..constants import MEASUREMENTS_FILE_BOXES
#     
#     m = Measurements()
#     m.load(MEASUREMENTS_FILE_BOXES)
#     m.discard_time()
#     deviation = m.deviations(minimum_measurements=minimum_measurements, return_as_map=True)[0]
#     deviation[deviation <= 0] = float('inf')
#     
#     if interpolate:
#         def interpolate(deviation, method):
#             ## check where data is
#             data_points = (np.where(np.isfinite(deviation)))
#             data_values = deviation[data_points]
#             
#             ## interpolate 
#             logger.debug('Interpolating deviation with method %s.', method)
#             
#             interpolated_points = (np.where(np.isinf(deviation)))
#             interpolated_values = scipy.interpolate.griddata(data_points, data_values, interpolated_points, method=method)
#             interpolated_values[np.logical_or(interpolated_values <= 0, np.logical_not(np.isfinite(interpolated_values)))] = float('inf')
#             
#             deviation[interpolated_points] = interpolated_values
#             
#             return deviation
#         
#         deviation = interpolate(deviation, 'linear')
#         deviation = interpolate(deviation, 'nearest')
#     
#     return deviation