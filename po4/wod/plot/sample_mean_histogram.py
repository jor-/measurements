from util.logging import Logger
from measurements.po4.wod.plot.interface import plot_mean_histogram


with Logger():
    plot_mean_histogram(x_min=0, x_max=15)