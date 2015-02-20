import measurements.land_sea_mask.data
import measurements.po4.wod.deviation.values
import util.logging

with util.logging.Logger():
    measurements.po4.wod.deviation.values.for_TMM(t_dim=12)
    