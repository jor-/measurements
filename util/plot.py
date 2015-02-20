import os.path

import numpy as np
import matplotlib.pyplot as plt

import util.plot



def distribution_time(measurement_dict, file='/tmp/distribution_time.png', time_step=1/1., t_min=None, t_max=None, tick_power=None):
    m = measurement_dict
    m.discard_space()
    m.categorize_indices((time_step,))
    
    n = m.numbers()
    t = n[:,0]
    if t_min is None:
        t_min = t.min()
    if t_max is None:
        t_max = t.max()
    n = n[:,4]
    
    util.plot.histogram(t, file, weights=n, bins=range(int(t_min),int(t_max+2)), tick_power=tick_power)


def distribution_year(measurement_dict, file='/tmp/distribution_year.png', time_step=1/365., tick_power=None):
    m = measurement_dict
    m.discard_space()
    m.discard_year()
    m.categorize_indices((time_step,))
    n = m.numbers()
    t = n[:,0] / time_step
    n = n[:,4]
    util.plot.histogram(t, file, weights=n, bins=range(0, int(np.ceil(1/time_step))+1), tick_power=tick_power)


def distribution_depth(measurement_dict, file='/tmp/distribution_depth.png', z_max=None, bin_size=50, use_log_scale=True):
    m = measurement_dict
    m.dicard_indices((0,1,2))
    n = m.numbers()
    z = n[:,3]
    if z_max is None:
        z_max = z.max()
    n = n[:,4]
    util.plot.histogram(z, file, weights=n, bins=np.arange(0, z_max+1, bin_size), use_log_scale=use_log_scale)


def value_histograms(measurement_dict, lsm, file='/tmp/value_histogram_{}.png', min_measurements=1, step_size=0.01):
    measurement_dict.transform_indices_to_lsm(lsm)
    n = measurement_dict.numbers(min_measurements=min_measurements)
    for i in range(len(n)):
        measurement_index = n[i,:-1]
        util.plot.histogram(measurement_dict[measurement_index], file.format(measurement_index), step_size=step_size)



def _prepare_filename(file, lsm=None, insertion=None):
    file_prefix, file_ext  = os.path.splitext(file)
    if insertion is not None:
        file_prefix = file_prefix + '_' + insertion
    if lsm is not None:
        file_prefix = file_prefix + '_' + str(lsm)
    
    return file_prefix + file_ext


def _plot_map(data, lsm, file, layer=None, v_min=None, v_max=None, use_log_scale=False):
    data = lsm.insert_index_values_in_map(data, no_data_value=np.inf)
    if layer is not None:
        data = data[:, :, :, layer]
        data = data.reshape(data.shape + (1,))
    file = _prepare_filename(file, lsm)
    util.plot.data(data, file, no_data_value=np.inf, vmin=v_min, vmax=v_max, use_log_scale=use_log_scale)
    
def _plot_histogram(data, lsm, file, bins=None, step_size=None, v_min=None, v_max=None, use_log_scale=False, tick_power=None):
#     file_prefix, file_ext  = os.path.splitext(file)
#     file = file_prefix + '_histogram_time_' + str(lsm.t_dim) +  file_ext
    file = _prepare_filename(file, lsm, 'histogram')
    util.plot.histogram(data, file, bins=bins, step_size=step_size, x_min=v_min, x_max=v_max, use_log_scale=use_log_scale, tick_power=tick_power)

def _plot(data, lsm, file, bins=None, step_size=None, layer=None, v_min=None, v_max=None, use_log_scale=False, type='both'):
    if type == 'both' or type == 'map':
        _plot_map(data, lsm, file, layer=layer, v_min=v_min, v_max=v_max, use_log_scale=use_log_scale)
    if type == 'both' or type == 'histogram':
        _plot_histogram(data[:,-1], lsm, file, bins=bins, step_size=step_size, v_min=v_min, v_max=v_max, use_log_scale=use_log_scale)


def distribution_space(measurement_dict, lsm, file='/tmp/distribution_space.png', layer=None, use_log_scale=True, type='both'):
    measurement_dict.transform_indices_to_lsm(lsm)
    data = measurement_dict.numbers()
#     _plot(data, lsm, file, bins=(10**0, 10**1, 10**2, 10**3, 10*4), v_min=None, v_max=None, use_log_scale=True, type=type)
    _plot_map(data, lsm, file, layer=layer, v_min=0, use_log_scale=use_log_scale)
    _plot_histogram(data[:,-1], lsm, file, step_size=1, v_min=1, v_max=200, use_log_scale=use_log_scale)

def sample_mean(measurement_dict, lsm, file='/tmp/sample_mean.png', vmax=None, layer=None, type='both'):
    measurement_dict.transform_indices_to_lsm(lsm)
    data = measurement_dict.means()
#     _plot(data, lsm, file, step_size=0.1, v_min=0, v_max=5, type=type)
    _plot_map(data, lsm, file, layer=layer, v_min=0, v_max=v_max)
    _plot_histogram(data[:,-1], lsm, file, step_size=0.1, v_min=0, v_max=4, tick_power=3)

def sample_deviation(measurement_dict, lsm, file='/tmp/sample_deviation.png', vmax=None, layer=None, type='both', min_measurements=5):
    measurement_dict.transform_indices_to_lsm(lsm)
    data = measurement_dict.deviations(minimum_measurements=min_measurements)
#     _plot(data, lsm, file, step_size=0.05, v_min=0, v_max=2, type=type)
    _plot_map(data, lsm, file, layer=layer, v_min=0, v_max=v_max)
    _plot_histogram(data[:,-1], lsm, file, step_size=0.05, v_min=0, v_max=2, use_log_scale=True)

