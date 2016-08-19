import numpy as np

import measurements.dop.pw.data
import measurements.po4.wod.data.values


def points_near_water_mask(lsm, max_land_boxes=0):
    return (measurements.dop.pw.data.points_near_water_mask(lsm, max_land_boxes=max_land_boxes), measurements.po4.wod.data.values.points_near_water_mask(lsm, max_land_boxes=max_land_boxes))


def points_near_water_mask_concatenated(lsm, max_land_boxes=0):
    return np.concatenate(points_near_water_mask(lsm, max_land_boxes=max_land_boxes))
