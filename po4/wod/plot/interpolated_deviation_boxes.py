import logging
logging.basicConfig(level=logging.DEBUG)

from util.plot import set_font_size
from interface import plot_interpolted_deviation_boxes as plot


set_font_size(size=20)
plot(year_len=12, vmin=0.05, vmax=0.5)