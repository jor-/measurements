import numpy as np

import measurements.dop.pw.data
import measurements.po4.wod.data.values

import measurements.dop.pw.deviation
import measurements.po4.wod.deviation.values




def points():
    return (measurements.dop.pw.data.points(), measurements.po4.wod.data.values.points())


def results():
    return (measurements.dop.pw.data.results(), measurements.po4.wod.data.values.results())


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

