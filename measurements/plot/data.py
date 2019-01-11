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


def _change_t_dim(data, new_t_dim=None):
    t_dim = data.shape[0]
    if new_t_dim is None:
        new_t_dim = t_dim
    factor = t_dim / new_t_dim
    if factor.is_integer() and factor >= 1:
        if factor > 1:
            factor = int(factor)
            new_shape = (new_t_dim,) + data.shape[1:]
            new_data = np.zeros(new_shape)
            for i in range(factor):
                new_data += data[i::factor]
            new_data /= factor
        else:
            new_data = data
    else:
        raise ValueError(f'Old time dim {t_dim} must be a mutiple of new time dim {new_t_dim}.')
    assert new_data.ndim == data.ndim
    assert new_data.shape[0] == new_t_dim
    return new_data


def plot_time_space_depth(data, file, v_max=None, overwrite=False, t_dim=None):
    assert data.ndim == 4
    data = _change_t_dim(data, new_t_dim=t_dim)
    v_min = 0
    contours = False
    # fix v_max if needed
    if v_max == 'fixed':
        v_max = util.plot.auxiliary.v_max(data)
    # prepare base file
    file = _append_to_filename(file, '_-_time_{time}_depth_{depth}')
    # plot data
    util.plot.save.data(file, data, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=not contours, overwrite=overwrite)


def plot_space_depth(data, file, v_max=None, overwrite=False):
    assert data.ndim == 4
    v_min = 0
    contours = False
    # average time_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data_time_averaged = np.nanmean(data, axis=0)
    # fix v_max if needed
    if v_max == 'fixed':
        v_max = util.plot.auxiliary.v_max(data)
    # prepare base file
    file = _append_to_filename(file, '_-_depth_{depth}')
    # plot time averaged
    util.plot.save.data(file, data_time_averaged, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=not contours, overwrite=overwrite)


def plot_depth(data, base_file, sample_lsm, v_max=None, overwrite=False):
    assert data.ndim == 4
    v_min = 0
    # prepare base file
    file = _append_to_filename(base_file, '_-_depth')
    # plot all averaged without depth
    if overwrite or not os.path.exists(file):
        dtype = np.float128
        volumes_map = sample_lsm.volumes_map(t_dim=None, dtype=dtype)
        weights_map = volumes_map / (data.shape[0] * np.nansum(volumes_map, dtype=dtype, axis=(0, 1)))
        data_averaged_all_without_depth = np.nansum(data * weights_map, dtype=dtype, axis=(0, 1, 2))
        assert data_averaged_all_without_depth.shape == (data.shape[3],)
        if v_max is None:
            v_max = util.plot.auxiliary.v_max(data_averaged_all_without_depth)
        util.plot.save.line(file, sample_lsm.z_center, data_averaged_all_without_depth, y_min=v_min, y_max=v_max, line_color='b', line_width=3, xticks=np.arange(5) * 2000, overwrite=overwrite)


def plot_time(data, base_file, sample_lsm, v_max=None, overwrite=False):
    assert data.ndim == 4
    v_min = 0
    # prepare base file
    file = _append_to_filename(base_file, '_-_time')
    # plot all averaged without depth
    if overwrite or not os.path.exists(file):
        # average
        dtype = np.float128
        weights_map = sample_lsm.normalized_volume_weights_map(t_dim=None, dtype=dtype)
        data_averaged = np.nansum(data * weights_map, dtype=dtype, axis=(1, 2, 3))
        # x values
        t_dim = data.shape[0]
        assert data_averaged.shape == (t_dim,)
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


def plot(data, base_file, sample_lsm, plot_type='all', v_max=None, overwrite=False):
    if plot_type == 'all':
        plot_time_space_depth(data, base_file, v_max=v_max, overwrite=overwrite)
        plot_space_depth(data, base_file, v_max=v_max, overwrite=overwrite)
        plot_depth(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
        plot_time(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
        plot_histogram(data, base_file, v_max=v_max, overwrite=overwrite)
        if v_max is None:
            # plot time difference
            diff = _data_abs_time_difference(data)
            diff_base_file = _append_to_filename(base_file, '_-_abs_time_diff')
            plot_space_depth(diff, diff_base_file, v_max=v_max, overwrite=overwrite)
            plot_depth(diff, diff_base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
            # plot with fixed v_max
            v_max = 'fixed'
            fixed_base_file = _append_v_max_to_filename(base_file, v_max)
            plot_time_space_depth(data, fixed_base_file, v_max=v_max, overwrite=overwrite)
            plot_space_depth(data, fixed_base_file, v_max=v_max, overwrite=overwrite)
            fixed_diff_base_file = _append_v_max_to_filename(diff_base_file, v_max)
            plot_space_depth(diff, fixed_diff_base_file, v_max=v_max, overwrite=overwrite)
    elif plot_type == 'time_space_depth':
        plot_time_space_depth(data, base_file, v_max=v_max, overwrite=overwrite)
    elif plot_type == 'space_depth':
        plot_space_depth(data, base_file, v_max=v_max, overwrite=overwrite)
    elif plot_type == 'depth':
        plot_depth(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
    elif plot_type == 'time':
        plot_time(data, base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
    elif plot_type == 'histogram':
        plot_histogram(data, base_file, v_max=v_max, overwrite=overwrite)
    elif plot_type == 'space_depth_of_time_diff' or plot_type == 'depth_of_time_diff':
        diff = _data_abs_time_difference(data)
        diff_base_file = _append_to_filename(base_file, '_-_abs_time_diff')
        if plot_type == 'space_depth_of_time_diff':
            plot_space_depth(diff, diff_base_file, v_max=v_max, overwrite=overwrite)
        if plot_type == 'depth_of_time_diff':
            plot_depth(diff, diff_base_file, sample_lsm, v_max=v_max, overwrite=overwrite)
    else:
        raise ValueError(f'Unknown plot type {plot_type}.')


# *** statistics (without correlation) *** #

def means_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('expected_value', 'mean'),
            kind_id=measurements_object.mean_id,
            plot_name='means_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.means_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


def quantiles_for_sample_lsm(measurements_object, quantile, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('expected_value', 'quantile'),
            kind_id=measurements_object.quantile_id(quantile, min_measurements=min_measurements),
            plot_name=f'concentration_quantiles_for_sample_lsm_{float(quantile):0<4}')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.quantiles_for_sample_lsm(quantile, min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


def concentration_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('dispersion', 'standard_deviation', 'concentration_standard_deviations'),
            kind_id=measurements_object.standard_deviation_id,
            plot_name='concentration_standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.concentration_standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


def average_noise_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('dispersion', 'standard_deviation', 'average_noise_standard_deviations'),
            kind_id=measurements_object.standard_deviation_id,
            plot_name='average_noise_standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.average_noise_standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


def standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('dispersion', 'standard_deviation', 'standard_deviations'),
            kind_id=measurements_object.standard_deviation_id,
            plot_name='standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.standard_deviations_for_sample_lsm()
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


def interquartile_range_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False):
    if file is None:
        kind_id = measurements_object.quantile_id(0, min_measurements=min_measurements)
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id.split(measurements.universal.constants.SEPERATOR)[1:])
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('dispersion', 'interquartile_range'),
            kind_id=kind_id,
            plot_name='interquartile_range_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.interquartile_range_for_sample_lsm(min_measurements=min_measurements)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


def relative_standard_deviations_for_sample_lsm(measurements_object, file=None, plot_type='all', v_max=None, overwrite=False):
    if file is None:
        # make kind id
        kind_id = measurements_object.mean_id.split(measurements.universal.constants.SEPERATOR)[:-1]
        kind_id += measurements_object.standard_deviation_id.split(measurements.universal.constants.SEPERATOR)[1:]
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        # make file name
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('dispersion', 'relative_standard_deviations'),
            kind_id=kind_id,
            plot_name='relative_standard_deviations_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.relative_standard_deviations_for_sample_lsm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.minimum(data, v_max)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


def quartile_coefficient_of_dispersion_for_sample_lsm(measurements_object, min_measurements=None, file=None, plot_type='all', v_max=None, overwrite=False):
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
            kind=os.path.join('dispersion', 'quartile_coefficient_of_dispersion'),
            kind_id=kind_id,
            plot_name='quartile_coefficient_of_dispersion_for_sample_lsm')
        file = _append_v_max_to_filename(file, v_max)
    data = measurements_object.quartile_coefficient_of_dispersion_for_sample_lsm(min_measurements=min_measurements)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.minimum(data, v_max)
    plot(data, file, measurements_object.sample_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite)


# *** correlation *** #

def sample_correlation_sparsity_pattern(measurements_object, file=None, permutation_method=None, overwrite=False):
    # set permutation method
    permutation_method_decomposition_correlation_old = measurements_object.permutation_method_decomposition_correlation
    measurements_object.permutation_method_decomposition_correlation = permutation_method
    # get file name
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('correlation', 'sample_correlation', 'sparsity_pattern'),
            kind_id=measurements_object.correlation_id,
            plot_name='sample_correlation_sparsity_pattern')
        file = file.replace('decomposition_{decomposition_type}{seperator}'.format(
            decomposition_type=measurements_object.decomposition_type_correlations,
            seperator=measurements.universal.constants.SEPERATOR), '')
    # plot if not existing
    if overwrite or not os.path.exists(file):
        # get data
        A = measurements_object.correlations_own_sample_matrix
        if permutation_method is not None:
            permutation_vector = measurements_object.correlations_own_permutation_vector
            A = A.tocoo(copy=False)
            A = matrix.permute.symmetric(A, permutation_vector)
        # plot
        util.plot.save.sparse_matrix_pattern(file, A, axis_labels=False)
    # reset old permutation method
    measurements_object.permutation_method_decomposition_correlation = permutation_method_decomposition_correlation_old


def sample_correlation_histogram(measurements_object, file=None, use_abs=False, overwrite=False):
    # get file name
    if file is None:
        if use_abs:
            plot_name = 'abs_sample_correlation_histogram'
        else:
            plot_name = 'sample_correlation_histogram'
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('correlation', 'sample_correlation', 'histogram'),
            kind_id=measurements_object.correlation_id,
            plot_name=plot_name)
        file = file.replace('decomposition_{decomposition_type}{seperator}'.format(
            decomposition_type=measurements_object.decomposition_type_correlations,
            seperator=measurements.universal.constants.SEPERATOR), '')
        file = file.replace('permutation_{permutation_method_decomposition_correlation}{seperator}'.format(
            permutation_method_decomposition_correlation=measurements_object.permutation_method_decomposition_correlation,
            seperator=measurements.universal.constants.SEPERATOR), '')
    # plot if not existing
    filename, file_extension = os.path.splitext(file)
    for use_log_scale in (False, True):
        file_with_scale = file.replace(file_extension, f'_log_scale_{use_log_scale}{file_extension}')
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
            kind=os.path.join('correlation', 'sample_correlation', 'sparsity_pattern'),
            kind_id=measurements_object.correlation_id,
            plot_name=correlation_and_sample_correlation_sparsity_pattern)
        file = file.replace('decomposition_{decomposition_type}{seperator}'.format(
            decomposition_type=measurements_object.decomposition_type_correlations,
            seperator=measurements.universal.constants.SEPERATOR), '')

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
