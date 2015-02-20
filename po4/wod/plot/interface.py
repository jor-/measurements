import numpy as np
import os.path

import matplotlib.pyplot as plt

from measurements.po4.wod.correlogram import estimation, model

import measurements.land_sea_mask.data
import measurements.po4.wod.mean.values
import measurements.po4.wod.deviation.values

import util.plot



# def interpolated_data(data_kind, lsm_kind, t_dim):
#     if data_kind == 'mean':
#         values = measurements.po4.wod.mean.values
#         
#         v_min = 0
#         v_max = 2.5
#         
#     elif data_kind == 'deviation':
#         values = measurements.po4.wod.deviation.values
#         
#         v_min = 0.05
#         if t_dim == 1:
#             v_max = 0.3
#         elif t_dim == 4:
#             v_max = 0.35
#         elif t_dim == 12:
#             v_max = 0.4
#         elif t_dim == 48:
#             v_max = 0.5
#     
#     
#     if lsm_kind == 'TMM':
#         data = values.for_TMM(t_dim=t_dim)
#     elif lsm_kind == 'WOA13':
#         data = values.for_WOA13(t_dim=t_dim)
#     elif lsm_kind == 'WOA13R':
#         data = values.for_WOA13R(t_dim=t_dim)
#     
#     file = '/tmp/po4_wod13_interpolated_{}_lsm_{}.png'.format(data_kind, lsm_kind.lower())
#     util.plot.data(data, file, no_data_value=np.inf, vmin=v_min, vmax=v_max)


def data(calculation_kind, data_kind, lsm_kind, t_dim):
    assert calculation_kind in ('sample', 'interpolated')
    assert data_kind in ('mean', 'deviation')
    assert lsm_kind in ('TMM', 'WOA13', 'WOA13R')
    
    if data_kind == 'mean':
        values = measurements.po4.wod.mean.values
        
        v_min = 0
        v_max = 2.5
        
    elif data_kind == 'deviation':
        values = measurements.po4.wod.deviation.values
        
        v_min = 0.05
        if t_dim == 1:
            v_max = 0.3
        elif t_dim == 4:
            v_max = 0.35
        elif t_dim == 12:
            v_max = 0.4
        elif t_dim == 48:
            v_max = 0.5
    
    if calculation_kind == 'sample':
        values = values.Interpolator()
        if lsm_kind == 'TMM':
            lsm = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=t_dim)
        elif lsm_kind == 'WOA13':
            lsm = measurements.land_sea_mask.data.LandSeaMaskWOA13(t_dim=t_dim)
        elif lsm_kind == 'WOA13R':
            lsm = measurements.land_sea_mask.data.LandSeaMaskWOA13R(t_dim=t_dim)
        data = values.sample_data_for_lsm(lsm)
        
    elif calculation_kind == 'interpolated':
        if lsm_kind == 'TMM':
            data = values.for_TMM(t_dim=t_dim)
        elif lsm_kind == 'WOA13':
            data = values.for_WOA13(t_dim=t_dim)
        elif lsm_kind == 'WOA13R':
            data = values.for_WOA13R(t_dim=t_dim)
    
    file = '/tmp/po4_wod13_{}_{}_lsm_{}.png'.format(calculation_kind, data_kind, lsm_kind.lower())
    util.plot.data(data, file, no_data_value=np.inf, vmin=v_min, vmax=v_max)



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
        file = os.path.join(path, 'po4_wod13_correlogram_min_measurements_' + str(min_measurements) + '_direction_' + str(direction_index) + '.png')
        plt.savefig(file, transparent=True)
        plt.close(fig)
        
        util.plot.trim(file)




def plot_interpolated_deviation_histogram(file='/tmp/po4_wod13_interpolated_deviation_histogram.png', step_size=0.01, x_min=None, x_max=None, use_log_scale=False):
    deviation = measurements.po4.wod.deviation.values.for_points()
    deviation[deviation < 0.05] = 0.051     # for rounding errors
    
    plot_histogram(deviation, file, step_size=step_size, x_min=x_min, x_max=x_max, use_log_scale=use_log_scale)
