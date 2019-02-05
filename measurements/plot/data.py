import os.path
import warnings

import numpy as np

import matrix.permute

import measurements.plot.constants
import measurements.universal.constants
import util.plot.save
import util.plot.auxiliary


def _append_to_filename(filename, suffix):
    file_root, file_extension = os.path.splitext(filename)
    filename = file_root + suffix + file_extension
    return filename


def _append_v_max_to_filename(filename, v_max):
    if v_max is not None:
        filename = _append_to_filename(filename, f'_-_v_max_{v_max}')
    return filename


def _data_abs_time_difference(data):
    diff = np.empty_like(data)
    m = data.shape[0] - 1
    diff[:m] = data[1:] - data[:m]
    diff[m] = data[0] - data[m]
    diff = np.abs(diff)
    return diff


def _data_abs_depth_difference(data):
    diff = np.full_like(data, np.nan)
    m = data.shape[-1] - 1
    diff[..., :m] = data[..., 1:] - data[..., :m]
    diff = np.abs(diff)
    return diff


def _change_t_dim(data, t_dim=None):
    old_t_dim = data.shape[0]
    if t_dim is None:
        t_dim = old_t_dim
    factor = old_t_dim / t_dim
    if factor.is_integer() and factor >= 1:
        if factor > 1:
            factor = int(factor)
            new_shape = (t_dim,) + data.shape[1:]
            new_data = np.zeros(new_shape)
            for i in range(factor):
                new_data += data[i::factor]
            new_data /= factor
        else:
            new_data = data
    else:
        raise ValueError(f'Old time dim {old_t_dim} must be a mutiple of new time dim {t_dim}.')
    assert new_data.ndim == data.ndim
    assert new_data.shape[0] == t_dim
    return new_data


def _average_data(data, sample_lsm, exclude_axis=None):
    dtype = np.float128
    if exclude_axis is None or exclude_axis == 0:
        # average all without time
        weights_map = sample_lsm.normalized_volume_weights_map(t_dim=None, dtype=dtype)
        data_averaged = np.nansum(data * weights_map, axis=(1, 2, 3), dtype=dtype)
        assert data_averaged.shape == (data.shape[0],)
        # average time
        if exclude_axis is None:
            data_averaged = np.nanmean(data_averaged)
    elif exclude_axis == 3:
        volumes_map = sample_lsm.volumes_map(t_dim=None, dtype=dtype)
        weights_map = volumes_map / (data.shape[0] * np.nansum(volumes_map, dtype=dtype, axis=(0, 1)))
        data_averaged = np.nansum(data * weights_map, axis=(0, 1, 2), dtype=dtype)
        assert data_averaged.shape == (data.shape[3],)
    else:
        raise ValueError(f'Unsupported exclude axis value {data_averaged}.')
    return data_averaged


def plot_time_space_depth(data, file, v_max=None, overwrite=False, colorbar=True, t_dim=None):
    assert data.ndim == 4
    data = _change_t_dim(data, t_dim=t_dim)
    v_min = 0
    # fix v_max if needed
    if v_max == 'fixed':
        v_max = util.plot.auxiliary.v_max(data)
    # prepare base file
    if not colorbar:
        file = _append_to_filename(file, '_-_no_colorbar')
    if data.shape[0] > 1:
        file = _append_to_filename(file, '_-_time_{time}_depth_{depth}')
    else:
        file = _append_to_filename(file, '_-_depth_{depth}')
    # plot data
    util.plot.save.data(file, data, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=False, colorbar=colorbar, overwrite=overwrite)


def plot_space_depth(data, file, v_max=None, overwrite=False, colorbar=True):
    plot_time_space_depth(data, file, v_max=v_max, overwrite=overwrite, t_dim=1, colorbar=colorbar)


def plot_depth(data, base_file, sample_lsm, v_max=None, overwrite=False, z_values='center'):
    assert data.ndim == 4
    v_min = 0
    # prepare base file
    file = _append_to_filename(base_file, '_-_depth')
    # plot all averaged without depth
    if overwrite or not os.path.exists(file):
        data_averaged = _average_data(data, sample_lsm, exclude_axis=3)
        if v_max is None:
            v_max = util.plot.auxiliary.v_max(data_averaged)
        if z_values == 'center':
            x = sample_lsm.z_center
        elif z_values == 'right':
            x = sample_lsm.z_right
        elif z_values == 'left':
            x = sample_lsm.z_left
        else:
            raise ValueError(f'Unknown z_values {z_values}. Only "center", "right" and "left" are supported.')
        util.plot.save.line(file, x, data_averaged, y_min=v_min, y_max=v_max, line_color='b', line_width=3, xticks=np.arange(5) * 2000, overwrite=overwrite)


def plot_time(data, base_file, sample_lsm, v_max=None, overwrite=False):
    assert data.ndim == 4
    v_min = 0
    # prepare base file
    file = _append_to_filename(base_file, '_-_time')
    # plot all averaged without depth
    if overwrite or not os.path.exists(file):
        data_averaged = _average_data(data, sample_lsm, exclude_axis=0)
        # x values
        t_dim = data.shape[0]
        x = np.arange(t_dim) / (t_dim - 1)
        # v max
        if v_max is None:
            v_max = util.plot.auxiliary.v_max(data_averaged)
        # plot
        util.plot.save.line(file, x, data_averaged, y_min=v_min, y_max=v_max, line_color='b', line_width=3, xticks=np.arange(5) / 4, overwrite=overwrite)


def plot_histogram(data, base_file, v_max=None, overwrite=False):
    assert data.ndim == 4
    v_min = 0
    v_max = util.plot.auxiliary.v_max(data)
    # prepare base file
    file = _append_to_filename(base_file, '_-_histogram')
    # plot histogram
    if overwrite or not os.path.exists(file):
        data_non_nan = data[~np.isnan(data)]
        if v_max is None:
            v_max = util.plot.auxiliary.v_max(data_non_nan)
        util.plot.save.histogram(file, data_non_nan, x_min=v_min, x_max=v_max, use_log_scale=True, overwrite=overwrite)


def plot(data, base_file, sample_lsm, plot_type='all', v_max=None, overwrite=False, **kargs):
    if plot_type == 'all':
        plot_time_space_depth(data, base_file, v_max=v_max, overwrite=overwrite)
        plot_space_depth(data, base_file, v_max=v_max, overwrite=overwrite)
        plot_depth(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
        plot_time(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
        plot_histogram(data, base_file, v_max=v_max, overwrite=overwrite)
        if v_max is None:
            # plot with fixed v_max
            fixed_base_file = _append_v_max_to_filename(base_file, 'fixed')
            plot_time_space_depth(data, fixed_base_file, v_max='fixed', overwrite=overwrite)
            plot_space_depth(data, fixed_base_file, v_max='fixed', overwrite=overwrite)
            # plot seasonal values
            if data.ndim == 4:
                t_dim = data.shape[0]
                if t_dim % 4 == 0:
                    seasonal_data = _change_t_dim(data, t_dim=4)
                    plot_time_space_depth(seasonal_data, base_file, v_max=None, overwrite=overwrite)
                    plot_time_space_depth(seasonal_data, base_file, v_max=None, overwrite=overwrite, colorbar=False)
                    plot_time_space_depth(seasonal_data, fixed_base_file, v_max='fixed', overwrite=overwrite)
                    plot_time_space_depth(seasonal_data, fixed_base_file, v_max='fixed', overwrite=overwrite, colorbar=False)
            # plot time difference
            diff = _data_abs_time_difference(data)
            diff_base_file = _append_to_filename(base_file, '_-_abs_time_diff')
            plot_space_depth(diff, diff_base_file, v_max=None, overwrite=overwrite)
            plot_depth(diff, diff_base_file, sample_lsm, v_max=None, overwrite=overwrite)
            fixed_diff_base_file = _append_v_max_to_filename(diff_base_file, 'fixed')
            plot_space_depth(diff, fixed_diff_base_file, v_max='fixed', overwrite=overwrite)
            # plot depth difference
            diff = _data_abs_depth_difference(data)
            diff_base_file = _append_to_filename(base_file, '_-_abs_depth_diff')
            plot_depth(diff, diff_base_file, sample_lsm, v_max=None, overwrite=overwrite, z_values='right')
    elif plot_type == 'time_space_depth':
        plot_time_space_depth(data, base_file, v_max=v_max, overwrite=overwrite, **kargs)
    elif plot_type == 'space_depth':
        plot_space_depth(data, base_file, v_max=v_max, overwrite=overwrite, **kargs)
    elif plot_type == 'depth':
        plot_depth(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite, **kargs)
    elif plot_type == 'time':
        plot_time(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite, **kargs)
    elif plot_type == 'histogram':
        plot_histogram(data, base_file, v_max=v_max, overwrite=overwrite, **kargs)
    elif plot_type == 'space_depth_of_time_diff' or plot_type == 'depth_of_time_diff':
        diff = _data_abs_time_difference(data)
        diff_base_file = _append_to_filename(base_file, '_-_abs_time_diff')
        if plot_type == 'space_depth_of_time_diff':
            plot_space_depth(diff, diff_base_file, v_max=v_max, overwrite=overwrite, **kargs)
        if plot_type == 'depth_of_time_diff':
            plot_depth(diff, diff_base_file, sample_lsm, v_max=v_max, overwrite=overwrite, **kargs)
    elif plot_type == 'depth_of_depth_diff':
        diff = _data_abs_depth_difference(data)
        diff_base_file = _append_to_filename(base_file, '_-_abs_depth_diff')
        plot_depth(diff, diff_base_file, sample_lsm, v_max=None, overwrite=overwrite, z_values='right')
    else:
        raise ValueError(f'Unknown plot type {plot_type}.')


# *** statistics (without correlation) *** #

def means_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kargs):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('expected_value', 'mean'),
            kind_id=measurements_object.mean_id,
            plot_name='means_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.means_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def concentration_quantiles_for_sample_lsm(measurements_object, quantile, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False, **kargs):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('expected_value', 'quantile'),
            kind_id=measurements_object.quantile_id(quantile, min_measurements=min_measurements),
            plot_name=f'concentration_quantiles_for_sample_lsm_{float(quantile):0<4}')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.concentration_quantiles_for_sample_lsm(quantile, min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def concentration_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kargs):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'standard_deviation', 'concentration_standard_deviations'),
            kind_id=measurements_object.standard_deviation_id,
            plot_name='concentration_standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.concentration_standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def average_noise_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kargs):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'standard_deviation', 'average_noise_standard_deviations'),
            kind_id=measurements_object.standard_deviation_id,
            plot_name='average_noise_standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.average_noise_standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kargs):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'standard_deviation', 'standard_deviations'),
            kind_id=measurements_object.standard_deviation_id,
            plot_name='standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def concentration_interquartile_range_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False, **kargs):
    if file is None:
        kind_id = measurements_object.quantile_id(0, min_measurements=min_measurements)
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id.split(measurements.universal.constants.SEPERATOR)[1:])
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'interquartile_range', 'concentration_interquartile_range'),
            kind_id=kind_id,
            plot_name='concentration_interquartile_range_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.concentration_interquartile_range_for_sample_lsm(min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def average_noise_interquartile_range_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False, **kargs):
    if file is None:
        kind_id = measurements_object.quantile_id(0, min_measurements=min_measurements)
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id.split(measurements.universal.constants.SEPERATOR)[1:])
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'interquartile_range', 'average_noise_interquartile_range'),
            kind_id=kind_id,
            plot_name='average_noise_interquartile_range_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.average_noise_interquartile_range_for_sample_lsm(min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def concentration_relative_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, data_max=None, overwrite=False, **kargs):
    if file is None:
        # make kind id
        kind_id = measurements_object.mean_id.split(measurements.universal.constants.SEPERATOR)[:-1]
        kind_id += measurements_object.standard_deviation_id.split(measurements.universal.constants.SEPERATOR)[1:]
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        # make file name
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'relative_standard_deviation', 'concentration_relative_standard_deviations'),
            kind_id=kind_id,
            plot_name='concentration_relative_standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
        if data_max is not None:
            file = _append_to_filename(file, f'_-_data_max_{data_max}')
    data = measurements_object.concentration_relative_standard_deviations_for_sample_lsm()
    if data_max is None:
        if v_max is not None and v_max != 'fixed':
            data_max = v_max
        else:
            data_max = util.plot.auxiliary.v_max(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data[np.abs(data) > data_max] = data_max
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def relative_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, data_max=None, overwrite=False, **kargs):
    if file is None:
        # make kind id
        kind_id = measurements_object.mean_id.split(measurements.universal.constants.SEPERATOR)[:-1]
        kind_id += measurements_object.standard_deviation_id.split(measurements.universal.constants.SEPERATOR)[1:]
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        # make file name
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'relative_standard_deviation', 'relative_standard_deviations'),
            kind_id=kind_id,
            plot_name='relative_standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
        if data_max is not None:
            file = _append_to_filename(file, f'_-_data_max_{data_max}')
    data = measurements_object.relative_standard_deviations_for_sample_lsm()
    if data_max is None:
        if v_max is not None and v_max != 'fixed':
            data_max = v_max
        else:
            data_max = util.plot.auxiliary.v_max(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data[np.abs(data) > data_max] = data_max
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


def concentration_quartile_coefficient_of_dispersion_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, data_max=None, overwrite=False, **kargs):
    if file is None:
        # make kind id
        try:
            len(min_measurements)
        except TypeError:
            kind_id = measurements_object.quantile_id(quantile=0, min_measurements=min_measurements)
        else:
            min_measurements_placeholder = 9876543210
            kind_id = measurements_object.quantile_id(quantile=0, min_measurements=min_measurements_placeholder)
            min_measurements_str = ','.join((str(int(m)) for m in min_measurements))
            kind_id = kind_id.replace(str(min_measurements_placeholder), min_measurements_str)
        kind_id = kind_id.split(measurements.universal.constants.SEPERATOR)
        kind_id = kind_id[1:]
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        # make file name
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('variability', 'quartile_coefficient_of_dispersion', 'concentration_quartile_coefficient_of_dispersion'),
            kind_id=kind_id,
            plot_name='concentration_quartile_coefficient_of_dispersion_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
        if data_max is not None:
            file = _append_to_filename(file, f'_-_data_max_{data_max}')
    data = measurements_object.concentration_quartile_coefficient_of_dispersion_for_sample_lsm(min_measurements=min_measurements)
    if data_max is None:
        if v_max is not None and v_max != 'fixed':
            data_max = v_max
        else:
            data_max = util.plot.auxiliary.v_max(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data[np.abs(data) > data_max] = data_max
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kargs)


# *** correlation *** #

def sample_correlation_sparsity_pattern(measurements_object, file=None, permutation_method=None, overwrite=False):
    # get file name
    if file is None:
        plot_name = f'sample_correlation_sparsity_pattern_-_permutation_{permutation_method}'
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('correlation', 'sample_correlation', 'sparsity_pattern'),
            kind_id=measurements_object.sample_correlation_id,
            plot_name=plot_name)
    # plot if not existing
    if overwrite or not os.path.exists(file):
        # get data
        A = measurements_object.correlations_own_sample_matrix
        if permutation_method is not None:
            permutation_method_decomposition_correlation_old = measurements_object.permutation_method_decomposition_correlation
            measurements_object.permutation_method_decomposition_correlation = permutation_method
            permutation_vector = measurements_object.correlations_own_permutation_vector
            measurements_object.permutation_method_decomposition_correlation = permutation_method_decomposition_correlation_old
            A = A.tocoo(copy=False)
            A = matrix.permute.symmetric(A, permutation_vector)
        # plot
        util.plot.save.sparse_matrix_pattern(file, A, axis_labels=False)


def sample_correlation_histogram(measurements_object, file=None, use_abs=False, overwrite=False):
    # get file name
    if file is None:
        plot_name = f'sample_correlation_histogram_-_abs_{use_abs}' + '_-_log_scale_{use_log_scale}'
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('correlation', 'sample_correlation', 'histogram'),
            kind_id=measurements_object.sample_correlation_id,
            plot_name=plot_name)
    # plot if not existing
    for use_log_scale in (False, True):
        file_with_scale = file.format(use_log_scale=use_log_scale)
        if overwrite or not os.path.exists(file_with_scale):
            # get data
            A = measurements_object.correlations_own_sample_matrix
            A.tocsc(copy=False)
            A.eliminate_zeros()
            data = A.data
            del A
            if use_abs:
                data = np.abs(data)
            # plot
            if use_abs:
                x_min = 0
                tick_number = 3
            else:
                x_min = -1
                tick_number = 5
            util.plot.save.histogram(file_with_scale, data, step_size=0.05, x_min=x_min, x_max=1, tick_number=tick_number, use_log_scale=use_log_scale)


def correlation_and_sample_correlation_sparsity_pattern(measurements_object, file=None, overwrite=False):
    # get file name
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('correlation', 'correlation', 'sparsity_pattern'),
            kind_id=measurements_object.correlation_id,
            plot_name='correlation_and_sample_correlation_sparsity_pattern')

    # plot if not existing
    if overwrite or not os.path.exists(file):
        # prepare data
        min_abs_value = 10**-4
        A = measurements_object.correlations_own_sample_matrix.tocsc(copy=False)
        A.data[np.abs(A.data) < min_abs_value] = 0
        B = measurements_object.correlations_own.tocsc(copy=False)
        B.data[np.abs(B.data) < min_abs_value] = 0
        # plot
        util.plot.save.sparse_matrices_patterns_with_differences(
            file, A, B,
            colors=((1, 0, 0), (1, 1, 1), (1, 0, 1), (0, 0, 1)),
            labels=('removed', 'inserted', 'changed', 'unchanged'),)
