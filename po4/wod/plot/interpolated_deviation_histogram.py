from util.logging import Logger
from measurements.po4.wod.plot.interface import plot_interpolated_deviation_histogram

with Logger():
    plot_interpolated_deviation_histogram(x_min=0, x_max=2)