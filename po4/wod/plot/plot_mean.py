from util.plot import set_font_size
from interface import plot_mean as plot

import logging
logging.basicConfig(level=logging.DEBUG)

set_font_size(size=20)
plot(year_len=52, discard_year=True, layer=0, vmax=2.0)