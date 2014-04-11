from util.plot import set_font_size
from interface import plot_sample_mean as plot

import logging
logging.basicConfig(level=logging.DEBUG)

set_font_size(size=20)
plot(vmax=2.0)