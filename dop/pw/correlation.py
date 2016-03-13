import measurements.util.data

import measurements.dop.pw.deviation
import measurements.dop.pw.constants as CONSTANTS
import measurements.po4.wod.deviation.values



def different_boxes_sample_covariances_dict_transformed(min_values, max_year_diff=float('inf'), sample_lsm=CONSTANTS.SAMPLE_LSM):
    return measurements.util.data.MeasurementsCovariance()


def same_box_sample_covariances(sample_lsm=CONSTANTS.SAMPLE_LSM):
    po4_concentration_covariances = measurements.po4.wod.deviation.values.concentration_deviation_for_points()**2
    po4_total_covariances = measurements.po4.wod.deviation.values.total_deviation_for_points()**2
    po4_same_box_correlation = po4_concentration_covariances / po4_total_covariances
    po4_average_same_box_correlation = po4_same_box_correlation.mean()

    points = measurements.dop.pw.data.points()
    deviations = measurements.dop.pw.deviation.total_deviation_for_points(points=points, sample_lsm=sample_lsm)
    covariances = deviations**2 * po4_average_same_box_correlation
    return covariances

