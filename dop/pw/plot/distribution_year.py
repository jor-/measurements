import measurements.dop.pw.data
import measurements.util.plot
from util.plot import set_font_size

import logging
logging.basicConfig(level=logging.DEBUG)


set_font_size(size=20)
measurements.util.plot.distribution_year(measurements.dop.pw.data.load_as_measurements(), file='/tmp/wod_po4_distribution_year.png', linewidth=4, spine_linewidth=2)