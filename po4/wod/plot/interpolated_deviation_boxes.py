from util.logging import Logger
from interface import plot_interpolted_deviation_boxes as plot


with Logger():
    plot(year_len=12, vmin=0.05, vmax=0.5)