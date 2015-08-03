import os.path
import numpy as np
import matplotlib.pyplot as pp

from . import  estimation, model


def make_correlation_plot(direction_index, discard_year=False, show_model=True, min_measurements=1):
    ## get model
    if show_model:
        correlation_model = model.Correlation_Model()
        model_f = lambda d: correlation_model.correlation_by_distance(d)
    else:
        model_f = None
    
    ## get estimated data
    direction = estimation.get_direction(direction_index, discard_year)
    shift = estimation.get_shift(direction_index, discard_year, min_measurements=min_measurements)
    correlation = estimation.get_correlation(direction_index, discard_year, min_measurements=min_measurements)
    number = estimation.get_number(direction_index, discard_year, min_measurements=min_measurements)
    x_max = shift[len(shift)-1]
    
    ## plot setup
    pp.figure(figsize=(15,10), dpi=150)
    pp.xlim((0, x_max))
    pp.ylim((-1, 1))
    pp.title('Direction: ' + str(direction))
    
    ## plot model
    if model_f is not None:
        x = np.arange(x_max)
        y = np.empty(x_max, np.float)
        for i in range(x_max):
            y[i] = model_f(direction * x[i])
        
        pp.plot(x, y, color='red', linewidth=3)
    
    ## plot estimated data
    size = np.log10(number ** 2) * 10
    pp.scatter(shift, correlation, size)
    
    


def show_correlation_plot(direction_index, discard_year=False, min_measurements=100, show_model=True):
    make_correlation_plot(direction_index, discard_year, min_measurements=1, show_model=show_model)
    pp.show()


def save_correlation_plots(path='/tmp/', discard_year=False, min_measurements=100, show_model=True):
    direction_indices = estimation.get_direction_indices(discard_year)
    
    for i in direction_indices:
        make_correlation_plot(i, discard_year, min_measurements=min_measurements, show_model=show_model)
        file = os.path.join(path, 'direction_' + str(i) + '.png')
        pp.savefig(file)