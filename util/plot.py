import numpy as np
import matplotlib.pyplot as plt

import measurements.util.map
import util.plot

from ndop.model.constants import (METOS_X_DIM as X_DIM, METOS_Y_DIM as Y_DIM, METOS_Z as Z_VALUES)


def distribution_space(measurement_dict, file='/tmp/distribution_space.png', use_log_norm=True):
    m = measurement_dict
    m.discard_time()
    m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
    data = measurements.util.map.insert_values_in_map(m.numbers(), no_data_value=np.inf)
    util.plot.data(data, file, no_data_value=np.inf, use_log_norm=use_log_norm)


def distribution_time(measurement_dict, file='/tmp/distribution_time.png', time_step=1/1., linewidth=2, spine_linewidth=2):
    m = measurement_dict
    m.discard_space()
    m.categorize_indices((time_step,))
    n = m.numbers()
    t = n[:,0]
    y = n[:,4]
    util.plot.line(t, y, file, linewidth=linewidth, ymin=0, xticks=range(1930, 2030, 20), spine_linewidth=spine_linewidth)


def distribution_year(measurement_dict, file='/tmp/distribution_year.png', time_step=1/365., linewidth=2, spine_linewidth=2):
    m = measurement_dict
    m.discard_space()
    m.discard_year()
    m.categorize_indices((time_step,))
    n = m.numbers()
    t = n[:,0] / time_step
    y = n[:,4]
    util.plot.line(t, y, file, linewidth=linewidth, ymin=0, spine_linewidth=spine_linewidth)


def distribution(measurement_dict, file='/tmp/distribution.png', year_len=12, use_log_norm=True):
    m = measurement_dict
    m.discard_year()
    m.categorize_indices((1./year_len,))
    m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
    data = measurements.util.map.insert_values_in_map(m.numbers(), no_data_value=np.inf)
    util.plot.data(data, file, no_data_value=np.inf, use_log_norm=use_log_norm)


def sample_mean(measurement_dict, file='/tmp/sample_mean.png', year_len=12, vmax=None, layer=None, discard_year=True):
    m = measurement_dict
    if discard_year:
        m.discard_year()
    m.categorize_indices((1./year_len,))
    m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
    data = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
    if layer is not None:
        data = data[:, :, :, layer]
        data = data.reshape(data.shape + (1,))
    util.plot.data(data, file, no_data_value=np.inf, vmin=0, vmax=vmax)


def sample_deviation(measurement_dict, file='/tmp/sample_deviation.png', year_len=12, vmax=None, layer=None, discard_year=True):
    m = measurement_dict
    if discard_year:
        m.discard_year()
    m.categorize_indices((1./year_len,))
    m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
    data = measurements.util.map.insert_values_in_map(m.deviations(minimum_measurements=5), no_data_value=np.inf)
    if layer is not None:
        data = data[:, :, :, layer]
        data = data.reshape(data.shape + (1,))