from util.logging import Logger

from measurements.po4.wod.data.io import load_measurement_dict_unsorted as load_measurement_dict
import measurements.util.plot


with Logger():
    measurements.util.plot.sample_mean(load_measurement_dict(), file='/tmp/wod_po4_sample_mean.png', year_len=12, discard_year=True, vmax=2.0)