import os.path

import numpy as np

import matrix.permute

import measurements.plot.constants
import measurements.universal.constants
import util.plot


def _values_for_sample_lsm(data, base_file, sample_lsm, overwrite=False):

    def _calculate_v_max(data):
        data = data[~np.isnan(data)]
        v_max = np.percentile(data, 99)
        v_max = np.round(v_max * 100) / 100
        return v_max
    v_min = 0
    contours = False

    file_root, file_extension = os.path.splitext(base_file)

    # plot data
    file_data = file_root + '_-_data' + file_extension
    v_max = _calculate_v_max(data)
    assert data.ndim == 4
    util.plot.data(file_data, data, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=not contours, overwrite=overwrite)

    # plot histogram
    file_histogram = file_root + '_-_histogram' + file_extension
    if overwrite or not os.path.exists(file_histogram):
        data_non_nan = data[~np.isnan(data)]
        util.plot.histogram(file_histogram, data_non_nan, x_min=v_min, x_max=v_max, use_log_scale=True, overwrite=overwrite)

    # plot time averaged
    file = file_root + '_-_time_averaged' + file_extension
    data_time_averaged = np.nanmean(data, axis=0)
    v_max = _calculate_v_max(data_time_averaged)
    assert data_time_averaged.ndim == 3
    util.plot.data(file, data_time_averaged, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=not contours, overwrite=overwrite)

    # plot all averaged without depth
    file = file_root + '_-_all_averaged_without_depth' + file_extension
    if overwrite or not os.path.exists(file):
        n = data.shape[3]
        data_averaged_all_without_depth = np.empty((n,), dtype=np.float128)
        volume_map = sample_lsm.volume_map
        for i in range(n):
            data_i = data[:, :, :, i]
            volume_map_i = volume_map[:, :, :, i]
            mask = ~np.isnan(data_i)
            data_i = data_i[mask]
            volume_map_i = volume_map_i[mask]
            data_averaged_i = np.sum(data_i * volume_map_i) / np.sum(volume_map_i)
            data_averaged_all_without_depth[i] = data_averaged_i
        v_max = _calculate_v_max(data_averaged_all_without_depth)
        util.plot.line(file, sample_lsm.z_center, data_averaged_all_without_depth, y_min=v_min, y_max=v_max, line_color='b', line_width=3, xticks=np.arange(5) * 2000, overwrite=overwrite)


def means_for_sample_lsm(measurements_object, file=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='mean',
            kind_id=measurements_object.mean_id,
            plot_name='means_for_sample_lsm')
    data = measurements_object.means_for_sample_lsm()
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm, overwrite=overwrite)


def quantiles_for_sample_lsm(measurements_object, quantile, min_measurements=None, file=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='quantile',
            kind_id=measurements_object.quantile_id(quantile, min_measurements=min_measurements),
            plot_name=f'concentration_quantiles_for_sample_lsm_{float(quantile):0<4}')
    data = measurements_object.quantiles_for_sample_lsm(quantile, min_measurements=min_measurements)
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm, overwrite=overwrite)


def concentration_standard_deviations_for_sample_lsm(measurements_object, file=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='standard_deviation',
            kind_id=measurements_object.standard_deviation_id,
            plot_name='concentration_standard_deviations_for_sample_lsm')
    data = measurements_object.concentration_standard_deviations_for_sample_lsm()
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm, overwrite=overwrite)


def standard_deviations_for_sample_lsm(measurements_object, file=None, overwrite=False):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='standard_deviation',
            kind_id=measurements_object.standard_deviation_id,
            plot_name='standard_deviations_for_sample_lsm')
    data = measurements_object.standard_deviations_for_sample_lsm()
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm, overwrite=overwrite)


def interquartile_range_for_sample_lsm(measurements_object, min_measurements=None, file=None, overwrite=False):
    if file is None:
        kind_id = measurements_object.quantile_id(0, min_measurements=min_measurements)
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id.split(measurements.universal.constants.SEPERATOR)[1:])
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='interquartile_range',
            kind_id=kind_id,
            plot_name='interquartile_range_for_sample_lsm')
    data = measurements_object.interquartile_range_for_sample_lsm(min_measurements=min_measurements)
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm, overwrite=overwrite)


def relative_standard_deviations_for_sample_lsm(measurements_object, max_value=2, file=None, overwrite=False):
    if file is None:
        kind_id = measurements_object.standard_deviation_id
        kind_id = kind_id.split(measurements.universal.constants.SEPERATOR)
        kind_id.append(f'max_value_{max_value:g}')
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='relative_standard_deviations',
            kind_id=kind_id,
            plot_name='relative_standard_deviations_for_sample_lsm')
    data = measurements_object.relative_standard_deviations_for_sample_lsm()
    data = np.minimum(data, max_value)
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm, overwrite=overwrite)


def quartile_coefficient_of_dispersion_for_sample_lsm(measurements_object, max_value=2, min_measurements=None, file=None, overwrite=False):
    if file is None:
        kind_id = measurements_object.quantile_id(0, min_measurements=min_measurements)
        kind_id = kind_id.split(measurements.universal.constants.SEPERATOR)
        kind_id = kind_id[1:]
        kind_id.append(f'max_value_{max_value:g}')
        kind_id = measurements.universal.constants.SEPERATOR.join(kind_id)
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='quartile_coefficient_of_dispersion',
            kind_id=kind_id,
            plot_name='quartile_coefficient_of_dispersion_for_sample_lsm')
    data = measurements_object.quartile_coefficient_of_dispersion_for_sample_lsm(min_measurements=min_measurements)
    data = np.minimum(data, max_value)
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm, overwrite=overwrite)


def sample_correlation_sparsity_pattern(measurements_object, file=None, permutation_method=None, overwrite=False):
    # set permutation method
    permutation_method_decomposition_correlation_old = measurements_object.permutation_method_decomposition_correlation
    measurements_object.permutation_method_decomposition_correlation = permutation_method
    # get file name
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('correlation',
                              'sample_correlation',
                              'sparsity_pattern'),
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
        util.plot.sparse_matrix_pattern(file, A, axis_labels=False)
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
            kind=os.path.join('correlation',
                              'sample_correlation',
                              'histogram'),
            kind_id=measurements_object.correlation_id,
            plot_name=plot_name)
        file = file.replace('decomposition_{decomposition_type}{seperator}'.format(
            decomposition_type=measurements_object.decomposition_type_correlations,
            seperator=measurements.universal.constants.SEPERATOR), '')
        file = file.replace('permutation_{permutation_method_decomposition_correlation}{seperator}'.format(
            permutation_method_decomposition_correlation=measurements_object.permutation_method_decomposition_correlation,
            seperator=measurements.universal.constants.SEPERATOR), '')
    # plot if not existing
    for use_log_scale in (False, True):
        file_with_scale = file.replace('.png', '_log_scale_{}.png'.format(use_log_scale))
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
            util.plot.histogram(file_with_scale, data, step_size=0.05, x_min=x_min, x_max=1, tick_number=tick_number, use_log_scale=use_log_scale)


def correlation_and_sample_correlation_sparsity_pattern(measurements_object, file=None, overwrite=False):
    # get file name
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=os.path.join('correlation',
                              'sample_correlation',
                              'sparsity_pattern'),
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
        util.plot.sparse_matrices_patterns_with_differences(
            file, A, B,
            colors=((1, 0, 0), (1, 1, 1), (1, 0, 1), (0, 0, 1)),
            labels=('removed', 'inserted', 'changed', 'unchanged'),)
