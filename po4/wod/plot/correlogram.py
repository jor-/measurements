from util.logging import Logger
from interface import plot_correlogram as plot

with Logger():
    for min_measurements in (10, 25, 50, 100, 200, 500):
        plot(show_model=False, min_measurements=min_measurements)