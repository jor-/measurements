import os.path
import warnings

import numpy as np
import matplotlib.pyplot as plt

import matrix.permute

import measurements.plot.util
import measurements.universal.constants
import util.plot.auxiliary
import util.plot.save


def _data_abs_time_difference(data, offset=1):
    diff = np.empty_like(data)
    offset = int(offset)
    m = len(data) - offset
    diff[:m] = data[offset:] - data[:m]
    diff[m:] = data[:offset] - data[m:]
    diff = np.abs(diff)
    return diff


def _data_abs_depth_difference(data):
    diff = np.full_like(data, np.nan)
    m = data.shape[-1] - 1
    diff[..., :m] = data[..., 1:] - data[..., :m]
    diff = np.abs(diff)
    return diff


def _change_t_dim(data, t_dim=None):
    if t_dim is not None:
        t_dim = int(t_dim)
        old_t_dim = data.shape[0]
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
    else:
        return data


def _average_data(data, sample_lsm, exclude_axes=None):
    dtype = np.float128
    if exclude_axes is None or exclude_axes == (0,):
        # average all without time
        weights_map = sample_lsm.normalized_volume_weights_map(t_dim=None, dtype=dtype)
        data_averaged = np.nansum(data * weights_map, axis=(1, 2, 3), dtype=dtype)
        assert data_averaged.shape == (data.shape[0],)
        # average time
        if exclude_axes is None:
            data_averaged = np.nanmean(data_averaged, dtype=dtype)
    elif exclude_axes == (0, 3) or exclude_axes == (3,):
        volumes_map = sample_lsm.volumes_map(t_dim=None, dtype=dtype)
        weights_map = volumes_map / np.nansum(volumes_map, dtype=dtype, axis=(0, 1))
        data_averaged = np.nansum(data * weights_map, axis=(1, 2), dtype=dtype)
        assert data_averaged.shape == (data.shape[0], data.shape[3])
        if exclude_axes == (3,):
            data_averaged = np.nanmean(data_averaged, axis=0, dtype=dtype)
            assert data_averaged.shape == (data.shape[3],)
    elif exclude_axes == (2, 3):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            data_averaged = np.nanmean(data, axis=(0, 1), dtype=dtype)
        assert data_averaged.shape == (data.shape[2], data.shape[3])
    else:
        raise ValueError(f'Unsupported exclude axes value {exclude_axes}.')
    return data_averaged


def plot_time_space_depth(data, file, v_max=None, t_dim=None, colorbar=True, overwrite=False, **kwargs):
    assert data.ndim == 4
    data = _change_t_dim(data, t_dim=t_dim)
    v_min = 0
    # prepare file
    file = measurements.plot.util.append_to_filename(file, '_-_time_space_depth')
    file = measurements.plot.util.append_v_max_to_filename(file, v_max)
    if data.shape[0] > 1:
        file = measurements.plot.util.append_to_filename(file, '_-_time_{time}_depth_{depth}')
    else:
        file = measurements.plot.util.append_to_filename(file, '_-_depth_{depth}')
    if not colorbar:
        file = measurements.plot.util.append_to_filename(file, '_-_no_colorbar')
    # fix v_max if needed
    if v_max == 'fixed':
        v_max = util.plot.auxiliary.v_max(data)
    # plot data
    util.plot.save.data(file, data, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=False, colorbar=colorbar, overwrite=overwrite, **kwargs)


def plot_space_depth(data, file, v_max=None, overwrite=False, colorbar=True, **kwargs):
    plot_time_space_depth(data, file, v_max=v_max, overwrite=overwrite, t_dim=1, colorbar=colorbar, **kwargs)


def plot_depth(data, file, sample_lsm, v_max=None, overwrite=False, depth_on_y_axis=True, depth_decimals=None, **kwargs):
    assert data.ndim == 4
    v_min = 0
    # prepare file
    file = measurements.plot.util.append_to_filename(file, '_-_depth')
    file = measurements.plot.util.append_v_max_to_filename(file, v_max)
    if not depth_on_y_axis:
        file = measurements.plot.util.append_to_filename(file, '_-_depth_on_x_axis')
    # plot all averaged without depth
    if overwrite or not os.path.exists(file):
        data_averaged = _average_data(data, sample_lsm, exclude_axes=(3,))
        # prepare x
        n = len(data_averaged)
        x = np.arange(0.5, n, 1)
        x[0] = 0
        x[-1] = n
        x_min = 0
        x_max = n
        # prepare depth tick labels
        transform_depth_tick = lambda ticks: _transform_depth_tick(ticks, sample_lsm, tick_decimals=depth_decimals, values='left')
        # plot
        if depth_on_y_axis:
            util.plot.save.fill_between_x(
                file, x, 0, data_averaged, y_min=x_min, y_max=x_max, x_min=v_min, x_max=v_max,
                color='b', overwrite=overwrite, tick_transform_y=transform_depth_tick, tick_interger_only_y=True, invert_y_axis=True,
                **kwargs)
        else:
            util.plot.save.fill_between(
                file, x, 0, data_averaged, x_min=x_min, x_max=x_max, y_min=v_min, y_max=v_max,
                color='b', overwrite=overwrite, tick_transform_x=transform_depth_tick, tick_interger_only_x=True,
                **kwargs)


def plot_time(data, file, sample_lsm, v_max=None, overwrite=False, **kwargs):
    assert data.ndim == 4
    v_min = 0
    # prepare file
    file = measurements.plot.util.append_to_filename(file, '_-_time')
    file = measurements.plot.util.append_v_max_to_filename(file, v_max)
    # plot all averaged without depth
    if overwrite or not os.path.exists(file):
        data_averaged = _average_data(data, sample_lsm, exclude_axes=(0,))
        # x values
        t_dim = data.shape[0]
        assert t_dim > 1
        x = np.arange(t_dim) / (t_dim - 1)
        # v max
        if v_max is None:
            v_max = util.plot.auxiliary.v_max(data_averaged)
        # plot
        util.plot.save.line(file, x, data_averaged, y_min=v_min, y_max=v_max, line_color='b', line_width=3, xticks=np.arange(5) / 4, overwrite=overwrite, **kwargs)


def plot_histogram(data, file, v_max=None, overwrite=False, **kwargs):
    assert data.ndim == 4
    v_min = 0
    # prepare file
    file = measurements.plot.util.append_to_filename(file, '_-_histogram')
    file = measurements.plot.util.append_v_max_to_filename(file, v_max)
    # plot histogram
    if overwrite or not os.path.exists(file):
        data_non_nan = data[~np.isnan(data)]
        if v_max is None:
            v_max = util.plot.auxiliary.v_max(data_non_nan)
        util.plot.save.histogram(file, data_non_nan, x_min=v_min, x_max=v_max, use_log_scale=True, overwrite=overwrite, **kwargs)


def _prepare_tick_lable(tick, tick_decimals=None):
    if tick_decimals is not None:
        tick_decimals = int(tick_decimals)
        tick_lable = np.around(tick, decimals=tick_decimals)
        if tick_decimals == 0:
            tick_lable = int(tick_lable)
    return tick_lable


def _transform_y_tick(tick, sample_lsm, tick_decimals=None):
    tick = tick / sample_lsm.y_dim * 180 - 90
    tick = _prepare_tick_lable(tick, tick_decimals=tick_decimals)
    if tick >= 0:
        tick_lable = '$' + str(tick) + '^{\\circ}$N'
    else:
        tick_lable = '$' + str(-tick) + '^{\\circ}$S'
    return tick_lable


def _transform_depth_tick(tick, sample_lsm, tick_decimals=None, values='center_with_bounds'):
    if np.issubdtype(type(tick), np.floating):
        assert tick.is_integer()
        tick = int(tick)
    assert np.issubdtype(type(tick), np.integer)
    if values == 'left':
        if 0 <= tick <= sample_lsm.z_dim - 1:
            return sample_lsm.z_left[tick]
        elif tick == sample_lsm.z_dim:
            return sample_lsm.z_right[-1]
        else:
            return np.iinfo(np.int32).min
    elif values == 'center_with_bounds':
        if tick == 0:
            return sample_lsm.z_left[0]
        elif tick == sample_lsm.z_dim - 1:
            return sample_lsm.z_right[-1]
        elif 0 < tick < sample_lsm.z_dim - 1:
            return sample_lsm.z_center[tick]
        else:
            return np.iinfo(np.int32).min
    else:
        raise ValueError(f'Unksupported values {values}.')

    return _prepare_tick_lable(tick, tick_decimals=tick_decimals)


def plot_y_z_profile(data, file, sample_lsm, v_max=None, x_coordinate_from=None, x_coordinate_to=None, remove_parts_without_data=False, colorbar=True, overwrite=False, tick_number_x=None, tick_number_y=None, x_ticks_decimals=None, y_ticks_decimals=None, **kwargs):
    # prepare file
    file = measurements.plot.util.append_to_filename(file, '_-_y_z_profile')
    file = measurements.plot.util.append_v_max_to_filename(file, v_max)
    if x_coordinate_from is not None:
        file = measurements.plot.util.append_to_filename(file, f'_-_x_from_{x_coordinate_from}')
        x_coordinate_from = float(x_coordinate_from)
    if x_coordinate_to is not None:
        file = measurements.plot.util.append_to_filename(file, f'_-_x_to_{x_coordinate_to}')
        x_coordinate_to = float(x_coordinate_to)
    remove_parts_without_data = bool(remove_parts_without_data)
    if remove_parts_without_data:
        file = measurements.plot.util.append_to_filename(file, '_-_parts_without_data_removed')
    if not colorbar:
        file = measurements.plot.util.append_to_filename(file, '_-_no_colorbar')

    if overwrite or not os.path.exists(file):
        # prepare data
        if x_coordinate_from is not None or x_coordinate_to is not None:
            data = sample_lsm.apply_ranges_to_mask(data, outside_value=np.nan, x_from=x_coordinate_from, x_to=x_coordinate_to)
        profile = _average_data(data, sample_lsm, exclude_axes=(2, 3))

        # v min and v_max
        v_min = 0
        if v_max is None:
            v_max = util.plot.auxiliary.v_max(profile)

        # set default number of ticks
        n, m = profile.shape
        if tick_number_x is None:
            tick_number_x = int(np.floor(n / 30))
        if tick_number_y is None:
            tick_number_y = int(np.floor(m / 5))

        # plot data
        def plot_function(fig):
            axes_image = plt.imshow(profile.T, aspect='equal', extent=(0, n, m, 0), vmin=v_min, vmax=v_max)
            axes = fig.gca()

            # set x and y limits
            if remove_parts_without_data:
                columns_with_values = np.where(np.any(np.logical_not(np.isnan(profile)), axis=1))[0]
                left = columns_with_values.min()
                right = columns_with_values.max()
            else:
                left = 0
                right = profile.shape[0]
            plt.xlim(left=left, right=right)
            plt.ylim(top=0, bottom=profile.shape[1])

        transform_y_tick = lambda ticks: _transform_y_tick(ticks, sample_lsm, tick_decimals=x_ticks_decimals)
        transform_depth_tick = lambda ticks: _transform_depth_tick(ticks, sample_lsm, tick_decimals=y_ticks_decimals, values='left')
        util.plot.auxiliary.generic(
            file, plot_function, colorbar=colorbar,
            tick_number_x=tick_number_x, tick_number_y=tick_number_y,
            tick_interger_only_x=True, tick_interger_only_y=True,
            tick_transform_x=transform_y_tick, tick_transform_y=transform_depth_tick,
            **kwargs)


def plot(data, file, sample_lsm, plot_type='all', v_max=None, overwrite=False, colorbar=True, **kwargs):
    if plot_type == 'all':
        plot_time_space_depth(data, file, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)
        plot_space_depth(data, file, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)
        plot_depth(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, **kwargs)
        plot_time(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, **kwargs)
        plot_histogram(data, file, v_max=v_max, overwrite=overwrite, **kwargs)
        plot_y_z_profile(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, colorbar=colorbar, x_coordinate_from=None, x_coordinate_to=None, **kwargs)
        plot_y_z_profile(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, colorbar=colorbar, x_coordinate_from=125, x_coordinate_to=290, **kwargs)
        plot_y_z_profile(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, colorbar=colorbar, x_coordinate_from=290, x_coordinate_to=20, **kwargs)
        if v_max is None:
            # plot with fixed v_max
            fixed_file = measurements.plot.util.append_v_max_to_filename(file, 'fixed')
            plot_time_space_depth(data, fixed_file, v_max='fixed', overwrite=overwrite, colorbar=colorbar, **kwargs)
            plot_space_depth(data, fixed_file, v_max='fixed', overwrite=overwrite, colorbar=colorbar, **kwargs)
            # plot seasonal values
            if data.ndim == 4:
                t_dim = data.shape[0]
                if t_dim % 4 == 0:
                    seasonal_data = _change_t_dim(data, t_dim=4)
                    plot_time_space_depth(seasonal_data, file, v_max=None, overwrite=overwrite, colorbar=colorbar, **kwargs)
                    plot_time_space_depth(seasonal_data, fixed_file, v_max='fixed', overwrite=overwrite, colorbar=colorbar, **kwargs)
            # plot time difference
            try:
                time_diff_offset = kwargs['time_diff_offset']
            except KeyError:
                time_diff_offset = 1
            else:
                del kwargs['time_diff_offset']
            diff = _data_abs_time_difference(data, offset=time_diff_offset)
            diff_file = measurements.plot.util.append_to_filename(file, f'_-_abs_time_diff_{time_diff_offset}')
            plot_space_depth(diff, diff_file, v_max=None, overwrite=overwrite, colorbar=colorbar, **kwargs)
            plot_depth(diff, diff_file, sample_lsm, v_max=None, overwrite=overwrite, **kwargs)
            fixed_diff_file = measurements.plot.util.append_v_max_to_filename(diff_file, 'fixed')
            plot_space_depth(diff, fixed_diff_file, v_max='fixed', overwrite=overwrite, colorbar=colorbar, **kwargs)
            # plot depth difference
            diff = _data_abs_depth_difference(data)
            diff_file = measurements.plot.util.append_to_filename(file, '_-_abs_depth_diff')
            plot_depth(diff, diff_file, sample_lsm, v_max=None, overwrite=overwrite, **kwargs)
    elif plot_type == 'time_space_depth':
        plot_time_space_depth(data, file, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)
    elif plot_type == 'space_depth':
        plot_space_depth(data, file, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)
    elif plot_type == 'depth':
        plot_depth(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, **kwargs)
    elif plot_type == 'time':
        plot_time(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, **kwargs)
    elif plot_type == 'histogram':
        plot_histogram(data, file, v_max=v_max, overwrite=overwrite, **kwargs)
    elif plot_type == 'plot_y_z_profile':
        plot_y_z_profile(data, file, sample_lsm, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)
    elif plot_type == 'space_depth_of_time_diff' or plot_type == 'depth_of_time_diff':
        try:
            time_diff_offset = kwargs['time_diff_offset']
        except KeyError:
            time_diff_offset = 1
        else:
            del kwargs['time_diff_offset']
        diff = _data_abs_time_difference(data, offset=time_diff_offset)
        diff_file = measurements.plot.util.append_to_filename(file, f'_-_abs_time_diff_{time_diff_offset}')
        if plot_type == 'space_depth_of_time_diff':
            plot_space_depth(diff, diff_file, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)
        if plot_type == 'depth_of_time_diff':
            plot_depth(diff, diff_file, sample_lsm, v_max=v_max, overwrite=overwrite, **kwargs)
    elif plot_type == 'depth_of_depth_diff':
        diff = _data_abs_depth_difference(data)
        diff_file = measurements.plot.util.append_to_filename(file, '_-_abs_depth_diff')
        plot_depth(diff, diff_file, sample_lsm, v_max=None, overwrite=overwrite, **kwargs)
    else:
        raise ValueError(f'Unknown plot type {plot_type}.')


# *** statistics (without correlation) *** #

def means_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('expected_value', 'mean')
        kind_id = measurements_object.mean_id
        plot_name = 'means_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    data = measurements_object.means_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def concentration_quantiles_for_sample_lsm(measurements_object, quantile, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('expected_value', 'quantile')
        kind_id = measurements_object.quantile_id(quantile, min_measurements=min_measurements)
        plot_name = f'concentration_quantiles_for_sample_lsm_{float(quantile):0<4}'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    data = measurements_object.concentration_quantiles_for_sample_lsm(quantile, min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def concentration_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'standard_deviation', 'concentration_standard_deviations')
        kind_id = measurements_object.standard_deviation_id
        plot_name = 'concentration_standard_deviations_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    data = measurements_object.concentration_standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def average_noise_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'standard_deviation', 'average_noise_standard_deviations')
        kind_id = measurements_object.standard_deviation_id
        plot_name = 'average_noise_standard_deviations_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    data = measurements_object.average_noise_standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'standard_deviation', 'standard_deviations')
        kind_id = measurements_object.standard_deviation_id
        plot_name = 'standard_deviations_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    data = measurements_object.standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def concentration_interquartile_range_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'interquartile_range', 'concentration_interquartile_range')
        kind_id = measurements_object.quantile_id(0, min_measurements=min_measurements)
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id.split(measurements.universal.constants.SEPERATOR)[1:])
        plot_name = 'concentration_interquartile_range_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    data = measurements_object.concentration_interquartile_range_for_sample_lsm(min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def average_noise_interquartile_range_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'interquartile_range', 'average_noise_interquartile_range')
        kind_id = measurements_object.quantile_id(0, min_measurements=min_measurements)
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id.split(measurements.universal.constants.SEPERATOR)[1:])
        plot_name = 'average_noise_interquartile_range_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
        file = measurements.plot.util.append_v_max_to_filename(file, v_max)
    data = measurements_object.average_noise_interquartile_range_for_sample_lsm(min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def concentration_relative_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, data_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'relative_standard_deviation', 'concentration_relative_standard_deviations')
        kind_id = measurements_object.mean_id.split(measurements.universal.constants.SEPERATOR)[:-1]
        kind_id += measurements_object.standard_deviation_id.split(measurements.universal.constants.SEPERATOR)[1:]
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        plot_name = 'concentration_relative_standard_deviations_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
        if data_max is not None:
            file = measurements.plot.util.append_to_filename(file, f'_-_data_max_{data_max}')
            data_max = float(data_max)
    data = measurements_object.concentration_relative_standard_deviations_for_sample_lsm()
    if data_max is None:
        if v_max is not None and v_max != 'fixed':
            data_max = v_max
        else:
            data_max = util.plot.auxiliary.v_max(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data[np.abs(data) > data_max] = data_max
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def relative_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, data_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'relative_standard_deviation', 'relative_standard_deviations')
        kind_id = measurements_object.mean_id.split(measurements.universal.constants.SEPERATOR)[:-1]
        kind_id += measurements_object.standard_deviation_id.split(measurements.universal.constants.SEPERATOR)[1:]
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        plot_name = 'relative_standard_deviations_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
        if data_max is not None:
            file = measurements.plot.util.append_to_filename(file, f'_-_data_max_{data_max}')
            data_max = float(data_max)
    data = measurements_object.relative_standard_deviations_for_sample_lsm()
    if data_max is None:
        if v_max is not None and v_max != 'fixed':
            data_max = v_max
        else:
            data_max = util.plot.auxiliary.v_max(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data[np.abs(data) > data_max] = data_max
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


def concentration_quartile_coefficient_of_dispersion_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, data_max=None, overwrite=False, **kwargs):
    if file is None:
        kind = os.path.join('variability', 'quartile_coefficient_of_dispersion', 'concentration_quartile_coefficient_of_dispersion')
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
        plot_name = 'concentration_quartile_coefficient_of_dispersion_for_sample_lsm'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
        if data_max is not None:
            file = measurements.plot.util.append_to_filename(file, f'_-_data_max_{data_max}')
            data_max = float(data_max)
    data = measurements_object.concentration_quartile_coefficient_of_dispersion_for_sample_lsm(min_measurements=min_measurements)
    if data_max is None:
        if v_max is not None and v_max != 'fixed':
            data_max = v_max
        else:
            data_max = util.plot.auxiliary.v_max(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data[np.abs(data) > data_max] = data_max
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, **kwargs)


# *** correlation *** #

def correlation_histogram(measurements_object, file=None, use_abs=False, use_sample_correlation=False, overwrite=False, **kwargs):
    if file is None:
        if use_sample_correlation:
            kind_id = measurements_object.sample_correlation_id
            kind_folder_name = 'sample_correlation'
        else:
            kind_id = measurements_object.correlation_id
            kind_folder_name = 'correlation'
        kind = os.path.join('correlation', kind_folder_name, 'histogram')
        plot_name = kind_folder_name + f'_histogram_-_abs_{use_abs}' + '_-_log_scale_{use_log_scale}'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    # plot if not existing
    for use_log_scale in (False, True):
        file_with_scale = file.format(use_log_scale=use_log_scale)
        if overwrite or not os.path.exists(file_with_scale):
            # get data
            if use_sample_correlation:
                A = measurements_object.correlations_own_sample_matrix
            else:
                A = measurements_object.correlations_own
            A.tocsc(copy=False)
            A.eliminate_zeros()
            data = A.data
            del A
            if use_abs:
                data = np.abs(data)
            # plot
            if use_abs:
                x_min = 0
            else:
                x_min = -1
            if use_log_scale:
                tick_number = None
            else:
                if use_abs:
                    tick_number = 3
                else:
                    tick_number = 5
            util.plot.save.histogram(file_with_scale, data, step_size=0.05, x_min=x_min, x_max=1, tick_number=tick_number, use_log_scale=use_log_scale, **kwargs)


def correlation_sparsity_pattern(measurements_object, file=None, permutation_method=None, use_sample_correlation=False, overwrite=False, **kwargs):
    # set permutation method
    permutation_method_decomposition_correlation_old = measurements_object.permutation_method_decomposition_correlation
    if permutation_method is not None:
        measurements_object.permutation_method_decomposition_correlation = permutation_method
    # get file name
    if file is None:
        if use_sample_correlation:
            kind_id = measurements_object.sample_correlation_id
            if permutation_method is not None:
                kind_id += f'_-_permutation_{permutation_method}'
            kind_folder_name = 'sample_correlation'
        else:
            kind_id = measurements_object.correlation_id
            kind_folder_name = 'correlation'
        plot_name = kind_folder_name + '_sparsity_pattern'
        kind = os.path.join('correlation', kind_folder_name, 'sparsity_pattern')
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)
    # plot correlation matrix sparsity pattern if not existing
    if overwrite or not os.path.exists(file):
        # get matrix
        if use_sample_correlation:
            permutation_vector = measurements_object.correlations_own_permutation_vector
            A = measurements_object.correlations_own_sample_matrix.tocoo(copy=False)
            A = matrix.permute.symmetric(A, permutation_vector)
        else:
            A = measurements_object.correlations_own
        # plot
        util.plot.save.sparse_matrix_pattern(file, A, axis_labels=False, **kwargs)
    # plot decomposition matrix sparsity pattern if not existing
    if not use_sample_correlation:
        file = measurements.plot.util.append_to_filename(file, '_-_L')
        if overwrite or not os.path.exists(file):
            A = measurements_object.correlations_own_decomposition.L
            util.plot.save.sparse_matrix_pattern(file, A, axis_labels=False, **kwargs)
    # restore permutation method
    measurements_object.permutation_method_decomposition_correlation = permutation_method_decomposition_correlation_old


def correlation_and_sample_correlation_sparsity_pattern(measurements_object, file=None, overwrite=False, **kwargs):
    # get file name
    if file is None:
        kind = os.path.join('correlation', 'correlation', 'sparsity_pattern')
        kind_id = measurements_object.correlation_id
        plot_name = 'correlation_and_sample_correlation_sparsity_pattern'
        file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)

    # plot if not existing
    if overwrite or not os.path.exists(file):
        # prepare data
        A = measurements_object.correlations_own_sample_matrix.tocsc(copy=False)
        A.eliminate_zeros()
        B = measurements_object.correlations_own.tocsc(copy=False)
        B.eliminate_zeros()
        # plot
        util.plot.save.sparse_matrices_patterns_with_differences(
            file, A, B,
            colors=((1, 0, 0), (1, 1, 1), (1, 0, 1), (0, 0, 1)),
            labels=('removed', 'inserted', 'changed', 'unchanged'),
            **kwargs)
