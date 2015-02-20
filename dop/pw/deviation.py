import numpy as np

import measurements.dop.pw.data
import measurements.land_sea_mask.data

import util.logging
logger = util.logging.get_logger()


def average():
    from measurements.dop.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE
    
    ## calculate deviations
    m = measurements.dop.pw.data.measurement_dict()
    sample_lsm = measurements.land_sea_mask.data.LandSeaMaskWOA13R(t_dim=52)
    m.categorize_indices_to_lsm(sample_lsm, discard_year=True)
    deviations = m.deviations(minimum_measurements=DEVIATION_MIN_MEASUREMENTS)
    
    ## apply lower value bound
    deviations[deviations[:, -1] < DEVIATION_MIN_VALUE, -1] = DEVIATION_MIN_VALUE
    
    ## average
    average_deviation = np.mean(deviations[:, -1])
    logger.debug('Got averaged DOP deviation {} from {} estimations.'.format(average_deviation, len(deviations)))
    
    return average_deviation


def for_points(points=None):
    from measurements.dop.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE
    
    ## load points if not passed
    if points is None:
        points = measurements.dop.pw.data.points_and_values()[0]
    
    ## calculate average deviation
    average_deviation = average()
    
    ## return deviation
    deviation = np.ones(len(points)) * average_deviation
    
    return deviation
