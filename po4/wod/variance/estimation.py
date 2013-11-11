import numpy as np
# import scipy.interpolate

import logging
logger = logging.getLogger(__name__)

from measurements.po4.wod.data.measurements import Measurements


def space_variance_from_measurements(separation_values, minimum_measurements=10):
    from ..constants import MEASUREMENTS_FILE_COORDINATES
    from .constants import SEPARATION_VALUES, X_RANGE
    
    m = Measurements()
    m.load(MEASUREMENTS_FILE_COORDINATES)
    m.discard_time()
    if separation_values is None:
        separation_values = SEPARATION_VALUES
    m.categorize_indices(separation_values, X_RANGE)
    variance = m.variances(minimum_measurements=minimum_measurements)
    
    # discard negative values
    variance = variance[variance[:, 4] > 0]
    # discard time
    variance = variance[:, 1:5]
    
    return variance


# def space_variance_from_measurements_by_boxes(minimum_measurements=10, interpolate=True):
#     from ..constants import MEASUREMENTS_FILE_BOXES
#     
#     m = Measurements()
#     m.load(MEASUREMENTS_FILE_BOXES)
#     m.discard_time()
#     variance = m.variances(minimum_measurements=minimum_measurements, return_as_map=True)[0]
#     variance[variance <= 0] = float('inf')
#     
#     if interpolate:
#         def interpolate(variance, method):
#             ## check where data is
#             data_points = (np.where(np.isfinite(variance)))
#             data_values = variance[data_points]
#             
#             ## interpolate 
#             logger.debug('Interpolating variance with method %s.', method)
#             
#             interpolated_points = (np.where(np.isinf(variance)))
#             interpolated_values = scipy.interpolate.griddata(data_points, data_values, interpolated_points, method=method)
#             interpolated_values[np.logical_or(interpolated_values <= 0, np.logical_not(np.isfinite(interpolated_values)))] = float('inf')
#             
#             variance[interpolated_points] = interpolated_values
#             
#             return variance
#         
#         variance = interpolate(variance, 'linear')
#         variance = interpolate(variance, 'nearest')
#     
#     return variance