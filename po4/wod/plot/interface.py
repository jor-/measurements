import numpy as np
import bisect
import os.path

import matplotlib.pyplot as plt

from measurements.po4.wod.data.results import Measurements_Unsorted as Measurements
from measurements.po4.wod.data.io import load_measurement_dict_unsorted as load_measurement_dict
from measurements.po4.wod.correlation import estimation, model
import measurements.po4.wod.deviation.io
import measurements.util.map
import ndop.oed.io
import util.plot

from ndop.model.constants import (METOS_X_DIM as X_DIM, METOS_Y_DIM as Y_DIM, METOS_Z as Z_VALUES)




def plot_interpolted_deviation_boxes(file='/tmp/wod_po4_box_deviation.png', year_len=12, vmin=0, vmax=None, layer=None, discard_year=True):
    data = ndop.oed.io.load_deviation_boxes()
    
    m = Measurements()
    m.add_results(data[:,:-1], data[:,-1])
    m.categorize_indices((1./year_len,))
    
    data = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
    if layer is not None:
        data = data[:, :, :, layer]
        data = data.reshape(data.shape + (1,))
    util.plot.data(data, file, no_data_value=np.inf, vmin=vmin, vmax=vmax)


def plot_correlogram(path='/tmp', show_model=True, min_measurements=1):
    
    direction_indices = estimation.get_direction_indices()
    
    ## get model
    if show_model:
        correlation_model = model.Correlation_Model()
        model_f = lambda d: correlation_model.correlation_by_distance(d)
    else:
        model_f = None
    
    for direction_index in direction_indices:
        ## get estimated data
        direction = estimation.get_direction(direction_index)
        shift = estimation.get_shift(direction_index, min_measurements=min_measurements)
        correlation = estimation.get_correlation(direction_index, min_measurements=min_measurements)
        number = estimation.get_number(direction_index, min_measurements=min_measurements)
        x_max = shift[len(shift)-1]
        
        ## plot setup
        fig = plt.figure(figsize=(15,10), dpi=150)
        plt.xlim((0, x_max))
        plt.ylim((-1, 1))
        
        title = ''
        if direction[0] > 0:
            days = int(int(np.round(1/direction[0])) / 365)
            if days == 1:
                title += '{} day '.format(days)
            else:
                title += '{} days '.format(days)
        if direction[1] > 0:
            if title != '':
                title += ', '
            title += '{}° longitude '.format(direction[1])
        if direction[2] > 0:
            if title != '':
                title += ', '
            title += '{}° latitude '.format(direction[2])
        if direction[3] > 0:
            if title != '':
                title += ', '
            title += '{}m depth'.format(direction[3])
        
        title = 'distance: ' + title
        plt.title(title)
        
        ## plot model
        if model_f is not None:
            x = np.arange(x_max)
            y = np.empty(x_max, np.float)
            for i in range(x_max):
                y[i] = model_f(direction * x[i])
            
            plt.plot(x, y, color='blue', linewidth=3)
        
        ## plot zero line
        plt.plot(np.arange(x_max+1), np.arange(x_max+1)*0, color='black', linewidth=2)
        
        ## plot estimated data
        size = np.log10(number ** 2) * 10
        plt.scatter(shift, correlation, size, color='red')
        
        ## set spine lines size
        util.plot.set_spine_line_size(fig, line_width=2)
        
        ## save plot
        file = os.path.join(path, 'wod_po4_correlogram_min_measurements_' + str(min_measurements) + '_direction_' + str(direction_index) + '.png')
        plt.savefig(file, transparent=True)
        plt.close(fig)
        
        util.plot.trim(file)




def plot_interpolated_deviation_histogram(file='/tmp/wod_po4_interpolated_deviation_histogram.png', step_size=0.01, x_min=None, x_max=None, use_log_scale=False):
    deviation = measurements.po4.wod.deviation.io.load_deviations()
    deviation[deviation < 0.05] = 0.051     # for rounding errors
    
    plot_histogram(deviation, file, step_size=step_size, x_min=x_min, x_max=x_max, use_log_scale=use_log_scale)


def plot_mean_histogram(file='/tmp/wod_po4_mean_histogram.png', step_size=0.01, x_min=None, x_max=None, use_log_scale=False):
    mean = measurements.po4.wod.data.io.load_measurement_results()
    
    plot_histogram(mean, file, step_size=step_size, x_min=x_min, x_max=x_max, use_log_scale=use_log_scale)



def plot_histogram(data, file, step_size=0.01, x_min=None, x_max=None, use_log_scale=False):
    if x_min is None:
        x_min = np.floor(np.min(data) / step_size) * step_size
    if x_max is None:
        x_max = np.ceil(np.max(data) / step_size) * step_size
    bins = np.arange(x_min, x_max+step_size, step_size)
    
    util.plot.histogram(data, bins, file, use_log_scale=use_log_scale)
