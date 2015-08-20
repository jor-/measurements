import numpy as np

import measurements.dop.pw.data
import measurements.land_sea_mask.data
import measurements.util.data

import util.logging
logger = util.logging.logger


def average():
    values = measurements.dop.pw.data.points_and_values()[1]
    average = values.mean()

    logger.debug('Got averaged DOP value {} from {} estimations.'.format(average, len(values)))
    return average


def for_points(points=None):
    from measurements.dop.constants import MEAN_MIN_MEASUREMENTS

    ## load points if not passed
    if points is None:
        points = measurements.dop.pw.data.points_and_values()[0]

    ## calculate average mean
    average_mean = average()

    ## calculate sample mean
    md = measurements.dop.pw.data.measurement_dict()
    sample_lsm = measurements.land_sea_mask.data.LandSeaMaskWOA13R(t_dim=48)
    md.categorize_indices_to_lsm(sample_lsm, discard_year=True)
    means_md = md.means(min_values=MEAN_MIN_MEASUREMENTS, return_type='measurements')

    ## convert points to sample lsm indices
    points_md = measurements.util.data.Measurements()
    n = len(points)
    points_md.append_values(points, (0,)*n)
    points_md.categorize_indices_to_lsm(sample_lsm, discard_year=True)
    points = points_md.keys()

    ## chose for each point sample mean or average mean
    means = np.empty(n, np.float64)

    for i in range(n):
        try:
            means[i] = means_md[points[i]][0]
        except KeyError:
            means[i] = average_mean

    return means
