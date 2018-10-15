import os.path

import numpy as np

import matrix.permute

import measurements.plot.constants
import measurements.universal.constants
import util.plot



def _values_for_sample_lsm(data, base_file, sample_lsm):
    def _calculate_v_max(data):
        data = data[~np.isnan(data)]
        v_max = np.percentile(data, 99)
        v_max = np.round(v_max * 100) / 100
        return v_max

    # plot data
    file_root, file_extension = os.path.splitext(base_file)
    file = file_root + '_-_data' + file_extension
    data_without_nans = data[~np.isnan(data)]
    v_min = 0
    v_max = _calculate_v_max(data_without_nans)
    contours = True
    util.plot.data(data, file, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=not contours)

    # plot histogram
    file_root, file_extension = os.path.splitext(file)
    file = file_root + '_-_histogram' + file_extension
    util.plot.histogram(data_without_nans, file, bins=9, x_min=v_min, x_max=v_max, use_log_scale=True)

    # plot time averaged
    file = file_root + '_-_time_averaged' + file_extension
    data_time_averaged = data.mean(axis=0)
    data_time_averaged_without_nans = data[~np.isnan(data_time_averaged)]
    v_max = _calculate_v_max(data_time_averaged_without_nans)
    util.plot.data(data, file, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=not contours)

    # plot all averaged without depth
    file = file_root + '_-_all_but_depth_averaged' + file_extension
    volume_map = sample_lsm.volume_map
    data_averaged_all_but_depth = np.nansum(data * volume_map, axis=(0, 1, 2)) / np.nansum(volume_map, axis=(0, 1, 2))
    v_max = _calculate_v_max(data_averaged_all_but_depth)
    util.plot.line(sample_lsm.z_center, data_averaged_all_but_depth, file, y_min=v_min, y_max=v_max, line_color='b', line_width=3, xticks=np.arange(5) * 2000)


def concentration_means_for_sample_lsm(measurements_object, file=None):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='mean',
            kind_id=measurements_object.mean_id,
            plot_name='concentration_means_for_sample_lsm')
    data = measurements_object.means_for_sample_lsm()
    # print if not existing
    if not os.path.exists(file):
        _values_for_sample_lsm(data, file, measurements_object.sample_lsm)


def concentration_standard_deviations_for_sample_lsm(measurements_object, file=None):
    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='standard_deviation',
            kind_id=measurements_object.standard_deviation_id,
            plot_name='concentration_standard_deviations_for_sample_lsm')
    data = measurements_object.concentration_standard_deviations_for_sample_lsm()
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm)


def sample_correlation_sparsity_pattern(measurements_object, file=None, permutation_method=None):
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
    if not os.path.exists(file):
        # get data
        A = measurements_object.correlations_own_sample_matrix
        if permutation_method is not None:
            permutation_vector = measurements_object.correlations_own_permutation_vector
            A = A.tocoo(copy=False)
            A = matrix.permute.symmetric(A, permutation_vector)
        # plot
        util.plot.spy(A, file, axis_labels=False)
    # reset old permutation method
    measurements_object.permutation_method_decomposition_correlation = permutation_method_decomposition_correlation_old


def sample_correlation_histogram(measurements_object, file=None, use_abs=False):
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
        if not os.path.exists(file_with_scale):
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
            util.plot.histogram(data, file_with_scale, step_size=0.05, x_min=x_min, x_max=1, tick_number=tick_number, use_log_scale=use_log_scale)


def correlation_and_sample_correlation_sparsity_pattern(measurements_object, file=None):
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
