from util.plot import set_font_size
from interface import plot_deviation as plot

import logging
logging.basicConfig(level=logging.DEBUG)

set_font_size(size=20)
plot(year_len=12, discard_year=True, vmax=0.5)
for i in range(15):
    plot(year_len=52, discard_year=True, vmax=0.5, layer=i)