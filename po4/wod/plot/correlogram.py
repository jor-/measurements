import logging
logging.basicConfig(level=logging.DEBUG)

from util.plot import set_font_size
from interface import plot_correlogram as plot

set_font_size(size=20)
for min_measurements in (10, 25, 50, 100, 200, 500):
    plot(show_model=False, min_measurements=min_measurements)