import measurements.dop.pw.data
import measurements.util.plot
from util.plot import set_font_size

import logging
logging.basicConfig(level=logging.DEBUG)


set_font_size(size=20)
measurements.util.plot.distribution(measurements.dop.pw.data.load_as_measurements(), file='/tmp/dop_pw_distribution.png', year_len=1, use_log_norm=False)