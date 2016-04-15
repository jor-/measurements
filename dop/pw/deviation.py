import numpy as np

import measurements.dop.constants
import measurements.dop.pw.constants
import measurements.dop.pw.data
import measurements.land_sea_mask.data

import util.logging
logger = util.logging.logger


def average_noise_deviation(sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    ## calculate deviations
    m = measurements.dop.pw.data.measurement_dict()
    m.categorize_indices_to_lsm(sample_lsm, discard_year=False)
    deviations = m.deviations(min_values=measurements.dop.constants.DEVIATION_MIN_MEASUREMENTS)
    
    if len(deviations) == 0:
        raise ValueError('No deviation available from data with smaple lsm {}.'.format(sample_lsm))

    ## apply lower value bound
    deviations[deviations[:, -1] < measurements.dop.constants.DEVIATION_MIN_VALUE, -1] = measurements.dop.constants.DEVIATION_MIN_VALUE

    ## average
    average_deviation = np.mean(deviations[:, -1])
    logger.debug('Got average DOP noise deviation {} from {} estimations.'.format(average_deviation, len(deviations)))

    return average_deviation


def average_concentration_deviation(sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    average_concentration_deviation = average_noise_deviation(sample_lsm=sample_lsm) * measurements.dop.pw.constants.DEVIATION_CONCENTRATION_NOISE_RATIO
    logger.debug('Got average DOP concentration deviation {} from concentration noise ratio {}.'.format(average_concentration_deviation, measurements.dop.pw.constants.DEVIATION_CONCENTRATION_NOISE_RATIO))
    return average_concentration_deviation


def average_total_deviation(sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    average_total_deviation = (average_noise_deviation(sample_lsm=sample_lsm)**2 + average_concentration_deviation(sample_lsm=sample_lsm)**2 )**(1/2)
    logger.debug('Got average DOP total deviation {}.'.format(average_total_deviation))
    return average_total_deviation


def total_deviation_for_points(points=None, sample_lsm=measurements.dop.pw.constants.SAMPLE_LSM):
    ## load points if not passed
    if points is None:
        points = measurements.dop.pw.data.points_and_results()[0]

    ## calculate total deviation
    total_deviation = np.ones(len(points)) * average_total_deviation(sample_lsm=sample_lsm)

    ## return deviation
    return total_deviation


