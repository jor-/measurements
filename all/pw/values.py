import numpy as np

import measurements.dop.pw.data
import measurements.po4.wod.data.values

import measurements.dop.pw.deviation
import measurements.po4.wod.deviation.values

import measurements.util.calculate


def points_and_results():
    (dop_points, dop_values) = measurements.dop.pw.data.points_and_values()
    po4_points = measurements.po4.wod.data.values.measurement_points()
    po4_values = measurements.po4.wod.data.values.measurement_results()

    points = (dop_points, po4_points)
    values = (dop_values, po4_values)

    return (points, values)


def points():
    return points_and_results()[0]


def results():
    return points_and_results()[1]


def deviation():
    dop_deviation = measurements.dop.pw.deviation.for_points()
    po4_deviation = measurements.po4.wod.deviation.values.for_points()

    deviation = (dop_deviation, po4_deviation)

    return deviation


def deviation_TMM(t_dim=12):
    lsm = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=t_dim)
    po4_deviation = measurements.po4.wod.deviation.values.for_TMM(t_dim=t_dim)
    dop_deviation = np.ones_like(po4_deviation) * measurements.dop.pw.deviation.average()

    ### concatenate
    deviation = np.concatenate([dop_deviation[np.newaxis, :], po4_deviation[np.newaxis, :]], axis=0)

    return deviation


# def sorted_indices():
#     unsorted_points = points()
#     sorted_indices = []
#     for p in unsorted_points:
#         sorted_indices.append(measurements.util.calculate.lex_sorted_indices(p))
#     return sorted_indices
