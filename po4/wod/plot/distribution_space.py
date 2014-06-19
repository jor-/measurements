from util.logging import Logger

from measurements.po4.wod.data.io import load_measurement_dict_unsorted as load_measurement_dict
import measurements.util.plot


with Logger():
    measurements.util.plot.distribution_space(load_measurement_dict(), file='/tmp/wod_po4_distribution_space.png')