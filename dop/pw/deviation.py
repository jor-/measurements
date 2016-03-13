import numpy as np

import measurements.dop.pw.constants
import measurements.dop.pw.data
import measurements.land_sea_mask.data

import util.logging
logger = util.logging.logger



def average_noise(sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    from measurements.dop.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE

    ## calculate deviations
    m = measurements.dop.pw.data.measurement_dict()
    m.categorize_indices_to_lsm(sample_lsm, discard_year=False)
    deviations = m.deviations(min_values=DEVIATION_MIN_MEASUREMENTS)
    
    if len(deviations) == 0:
        raise ValueError('No deviation available from data with smaple lsm {}.'.format(sample_lsm))

    ## apply lower value bound
    deviations[deviations[:, -1] < DEVIATION_MIN_VALUE, -1] = DEVIATION_MIN_VALUE

    ## average
    average_deviation = np.mean(deviations[:, -1])
    logger.debug('Got averaged noise DOP deviation {} from {} estimations.'.format(average_deviation, len(deviations)))

    return average_deviation


def average_total_deviation(sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    return average_noise(sample_lsm=sample_lsm)


def noise_deviation_for_points(points=None, sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    from measurements.dop.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE

    ## load points if not passed
    if points is None:
        points = measurements.dop.pw.data.points_and_results()[0]

    ## calculate average deviation
    average_noise_deviation = average_noise(sample_lsm=sample_lsm)

    ## return deviation
    noise_deviation = np.ones(len(points)) * average_noise_deviation

    return noise_deviation



def total_deviation_for_points(points=None, sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    return noise_deviation_for_points(points=points, sample_lsm=sample_lsm)


