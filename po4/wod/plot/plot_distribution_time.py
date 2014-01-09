from util.plot import set_font_size
from interface import plot_distribution_time as plot

import logging
logging.basicConfig(level=logging.DEBUG)

set_font_size(size=20)
plot(linewidth=4)