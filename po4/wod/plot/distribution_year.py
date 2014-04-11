import logging
logging.basicConfig(level=logging.DEBUG)

from measurements.po4.wod.data.io import load_measurement_dict_unsorted as load_measurement_dict
from measurements.util.plot
from util.plot import set_font_size


set_font_size(size=20)
measurements.util.plot.distribution_space(load_measurement_dict(), file='/tmp/wod_po4_distribution_year.png', linewidth=4, spine_linewidth=2)