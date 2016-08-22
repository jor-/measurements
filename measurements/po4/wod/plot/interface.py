import numpy as np
import os.path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d     # for projection='3d'

from measurements.po4.wod.correlogram import estimation, model

import measurements.land_sea_mask.lsm
import measurements.po4.wod.mean.values
import measurements.po4.wod.mean.constants
import measurements.po4.wod.deviation.values
import measurements.po4.wod.deviation.constants

import util.plot



def data(calculation_kind, data_kind, lsm_kind, t_dim, dir='/tmp'):
    assert calculation_kind in ('sample', 'interpolated')
    assert data_kind in ('mean', 'deviation', 'total_deviation', 'average_noise_deviation', 'concentration_deviation')
    assert lsm_kind in ('TMM', 'WOA13', 'WOA13R')
    default_t_dim = 12
    assert t_dim in (1, default_t_dim)


    ## chose lsm
    if lsm_kind == 'TMM':
        sample_lsm = measurements.land_sea_mask.lsm.LandSeaMaskTMM(t_dim=default_t_dim)
    elif lsm_kind == 'WOA13':
        sample_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13(t_dim=default_t_dim)
    elif lsm_kind == 'WOA13R':
        sample_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R(t_dim=default_t_dim)

    ## chose interpolator and min and max values
    if 'mean' in data_kind:
        values = measurements.po4.wod.mean.values
        constants = measurements.po4.wod.mean.constants
        min_values = constants.MIN_MEASUREMENTS

        v_min = 0
        v_max = 2.5

        histogram_step_size = 0.1
        histogram_v_max = 10
        depth_v_max = 2.5
        
        interpolator = values.InterpolatorConcentration(sample_lsm=sample_lsm, min_values=min_values)

    elif 'deviation' in data_kind:
        values = measurements.po4.wod.deviation.values
        constants = measurements.po4.wod.deviation.constants
        min_values = constants.MIN_MEASUREMENTS

        v_min = 0
        # if t_dim == 1:
        #     v_max = 0.3
        # elif t_dim == 4:
        #     v_max = 0.35
        # elif t_dim == 12:
        #     v_max = 0.4
        # elif t_dim == 48:
        #     v_max = 0.5
        v_max = 0.5

        histogram_step_size = 0.05
        histogram_v_max = 3.5
        depth_v_max = 0.3
        
        if data_kind == 'average_noise_deviation':
            interpolator = values.InterpolatorAverageNoise(sample_lsm=sample_lsm, min_values=min_values)
        elif data_kind == 'concentration_deviation':
            interpolator = values.InterpolatorConcentration(sample_lsm=sample_lsm, min_values=min_values)
        else:
            interpolator = values.InterpolatorTotal(sample_lsm=sample_lsm, min_values=min_values)

    ## chose date
    if calculation_kind == 'sample':
        data = interpolator.sample_data_for_lsm(sample_lsm)
    elif calculation_kind == 'interpolated':
        data = interpolator.interpolated_data_for_lsm(sample_lsm)
        volume_map = sample_lsm.volume_map
        data_depth_mean = np.nansum(data * volume_map, axis=(0,1,2)) / np.nansum(volume_map, axis=(0,1,2))
    
    contours = calculation_kind == 'interpolated'
    
    if t_dim == 1:
        data = data.mean(axis=0)

    ## prepare file names
    file_prefix = os.path.join(dir, 'po4_-_wod13_-_{data_kind}_-_{calculation_kind}_-_{sample_lsm}_-_min_values_{min_values}_-'.format(data_kind=data_kind, calculation_kind=calculation_kind, sample_lsm=sample_lsm, min_values=min_values))
    file_data = file_prefix + '.png'
    file_histogram = file_prefix + '_{}_histogram.png'.format(t_dim)
    file_depth_mean = file_prefix + '_depth_mean.png'.format()

    ## plot
    util.plot.data(data, file_data, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=not contours)
    util.plot.histogram(data[~np.isnan(data)], file_histogram, step_size=histogram_step_size, x_min=v_min, x_max=histogram_v_max, use_log_scale=True)
    if calculation_kind == 'interpolated':
        util.plot.line(sample_lsm.z_center, data_depth_mean, file_depth_mean, y_min=v_min, y_max=depth_v_max, line_color='b', line_width=3, xticks=np.arange(5)*2000)



def stationary_correlation(path='/tmp'):
    from measurements.constants import BASE_DIR
    c = np.load(os.path.join(BASE_DIR, 'po4/wod13/analysis/correlation/05_min_measurements_categorized_unsorted_correlations_stationary.npy'))

    dims = 4
    base_mask = np.all(np.isfinite(c), axis=1)

    ## plot correlation for changes in one and in two dims
    indices_list = [[(i,) for i in range(dim)], [(i,j) for i in range(dim) for j in range(dim) if i<j]]
    for indices in indices_list:
        for i in indices:
            ## calculate mask where only index i is not zero
            mask = base_mask.copy()
            for j in range(dims):
                if j not in i:
                    mask = np.logical_and(mask, c[:,j] == 0)

            ## scatter plot
            file = os.path.join(path, 'correlation_for_index_{}.png'.format(i))
            util.plot.scatter(c[mask][:,i], c[mask][:,-1], file, point_size=c[mask][:,-2])




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


