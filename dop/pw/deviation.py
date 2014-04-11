import numpy as np
import logging

import measurements.dop.pw.data


def get_deviation(points=None):
    from measurements.dop.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE
    from .constants import DEVIATION_SEPARATION_VALUES
    
    # load points if not passed
    if points is None:
        (points, values) = measurements.dop.pw.data.load_points_and_values()
    
    # calculate deviations
    m = measurements.dop.pw.data.load_as_measurements()
    m.discard_year()
    m.categorize_indices(DEVIATION_SEPARATION_VALUES)
    deviations = m.deviations(minimum_measurements=DEVIATION_MIN_MEASUREMENTS)
    
    # apply lower value bound
    deviations[deviations[:, -1] < DEVIATION_MIN_VALUE, -1] = DEVIATION_MIN_VALUE
    
    # average
    average_deviation = np.mean(deviations[:, -1])
    logging.debug('Got averaged DOP deviation {} from {} estimations.'.format(average_deviation, len(deviations)))
    
    # return deviation
    deviation = np.ones(len(points)) * average_deviation
    
    return deviation


# def for_points(points):
#     deviation = np.ones(len(points))
#     deviation *= get_average()
#     
#     return deviation