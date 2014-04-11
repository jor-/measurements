import numpy as np
import bisect
import os.path

import matplotlib.pyplot as plt

from measurements.po4.wod.data.results import Measurements_Unsorted as Measurements
from measurements.po4.wod.data.io import load_measurement_dict_unsorted as load_measurement_dict
from measurements.po4.wod.correlation import estimation, model
import measurements.util.map
import ndop.oed.io
import util.plot

from ndop.model.constants import (METOS_X_DIM as X_DIM, METOS_Y_DIM as Y_DIM, METOS_Z as Z_VALUES)


# def plot_distribution_space(file='/tmp/wod_po4_distribution_space.png'):
#     m = load_measurement_dict()
#     m.discard_time()
#     m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
#     data = measurements.util.map.insert_values_in_map(m.numbers(), no_data_value=np.inf)
#     util.plot.data(data, file, no_data_value=np.inf, use_log_norm=True)
# 
# 
# def plot_distribution_time(file='/tmp/wod_po4_distribution_time.png', time_step=1/1., linewidth=2, spine_linewidth=2):
#     m = load_measurement_dict()
#     m.discard_space()
#     m.categorize_indices((time_step,))
#     n = m.numbers()
#     t = n[:,0]
#     y = n[:,4]
#     util.plot.line(t, y, file, linewidth=linewidth, ymin=0, xticks=range(1930, 2030, 20), spine_linewidth=spine_linewidth)
# 
# 
# def plot_distribution_year(file='/tmp/wod_po4_distribution_year.png', time_step=1/365., linewidth=2, spine_linewidth=2):
#     m = load_measurement_dict()
#     m.discard_space()
#     m.discard_year()
#     m.categorize_indices((time_step,))
#     n = m.numbers()
#     t = n[:,0] / time_step
#     y = n[:,4]
#     util.plot.line(t, y, file, linewidth=linewidth, ymin=0, spine_linewidth=spine_linewidth)
# 
# 
# def plot_distribution(file='/tmp/wod_po4_distribution.png', time_len=12):
#     m = load_measurement_dict()
#     m.discard_year()
#     m.categorize_indices((1./time_len,))
#     m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
#     data = measurements.util.map.insert_values_in_map(m.numbers(), no_data_value=np.inf)
#     util.plot.data(data, file, no_data_value=np.inf, use_log_norm=True)
# 
# 
# def plot_sample_mean(file='/tmp/wod_po4_sample_mean.png', year_len=12, vmax=None, layer=None, discard_year=True):
#     m = load_measurement_dict()
#     if discard_year:
#         m.discard_year()
#     m.categorize_indices((1./year_len,))
#     m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
#     data = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
#     if layer is not None:
#         data = data[:, :, :, layer]
#         data = data.reshape(data.shape + (1,))
#     util.plot.data(data, file, no_data_value=np.inf, vmin=0, vmax=vmax)
# 
# 
# def plot_sample_deviation(file='/tmp/wod_po4_sample_deviation.png', year_len=12, vmax=None, layer=None, discard_year=True):
#     m = load_measurement_dict()
#     if discard_year:
#         m.discard_year()
#     m.categorize_indices((1./year_len,))
#     m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
#     data = measurements.util.map.insert_values_in_map(m.deviations(minimum_measurements=5), no_data_value=np.inf)
#     if layer is not None:
#         data = data[:, :, :, layer]
#         data = data.reshape(data.shape + (1,))
#     util.plot.data(data, file, no_data_value=np.inf, vmin=0, vmax=vmax)


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
        util.plot.change_spine_line_size(fig, linewidth=2)
        
        ## save plot
        file = os.path.join(path, 'wod_po4_correlogram_min_measurements_' + str(min_measurements) + '_direction_' + str(direction_index) + '.png')
        plt.savefig(file, transparent=True)
        plt.close(fig)
        
        util.plot.trim(file)
