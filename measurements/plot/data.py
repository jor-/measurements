import os.path

import numpy as np

import matrix.permute

import measurements.plot.constants
import measurements.universal.constants
import util.plot


def _calculate_v_max(data):
    v_max = np.percentile(data, 99)
    v_max = np.round(v_max * 100) / 100
    return v_max


def _values_for_sample_lsm(data, base_file, sample_lsm):
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


def _get_file_name(measurements_object, base_file=None, kind=None, plot_name=None):
    if base_file is None:
        if kind is None:
            kind = ''
        elif kind == 'mean':
            try:
                kind_id = measurements_object.mean_id
            except AttributeError:
                kind_id = ''
        elif kind == 'standard_deviation':
            try:
                kind_id = measurements_object.standard_deviation_id
            except AttributeError:
                kind_id = ''
        elif kind == 'correlation':
            try:
                kind_id = measurements_object.correlation_id
            except AttributeError:
                kind_id = ''
        else:
            raise ValueError('Unknown kind {}.'.format(kind))

        if plot_name is None:
            plot_name = ''

        base_file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind=kind,
            kind_id=kind_id,
            plot_name=plot_name)

    return base_file


def concentration_means_for_sample_lsm(measurements_object, base_file=None):
    data = measurements_object.means_for_sample_lsm
    file = _get_file_name(measurements_object,
                          base_file=base_file,
                          kind='mean',
                          plot_name='concentration_means_interpolated')
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm)


def concentration_standard_deviations_for_sample_lsm(measurements_object, base_file=None):
    data = measurements_object.concentration_standard_deviations_for_sample_lsm
    file = _get_file_name(measurements_object,
                          base_file=base_file,
                          kind='standard_deviation',
                          plot_name='concentration_standard_deviations_for_sample_lsm')
    _values_for_sample_lsm(data, file, measurements_object.sample_lsm)


def sample_correlation_sparsity_pattern(measurements_object, base_file=None, permutation_method=None):
    # get data
    A = measurements_object.correlations_own_sample_matrix
    permutation_vector = matrix.permute.permutation_vector(A, permutation_method)
    A = matrix.permute.symmetric(A, permutation_vector)
    # get file name
    permutation_method_decomposition_correlation_old = measurements_object.permutation_method_decomposition_correlation
    measurements_object.permutation_method_decomposition_correlation = permutation_method
    file = _get_file_name(measurements_object,
                          base_file=base_file,
                          kind='correlation',
                          plot_name='sample_correlation_sparsity_pattern')
    file = file.replace('decomposition_{decomposition_type}{seperator}'.format(
        decomposition_type=measurements_object.decomposition_type_correlations,
        seperator=measurements.universal.constants.SEPERATOR), '')
    measurements_object.permutation_method_decomposition_correlation = permutation_method_decomposition_correlation_old
    # plot
    util.plot.spy(A, file, axis_labels=False)


def sample_correlation_histogram(measurements_object, base_file=None, use_abs=False):
    # get data
    A = measurements_object.correlations_own_sample_matrix
    A.tocsc(copy=False)
    A.eliminate_zeros()
    data = A.data
    if use_abs:
        data = np.abs(data)
    # get file name
    plot_name = 'sample_correlation_histogram'
    if use_abs:
        plot_name = 'abs_' + plot_name
    file = _get_file_name(measurements_object,
                          base_file=base_file,
                          kind='correlation',
                          plot_name=plot_name)
    file = file.replace('decomposition_{decomposition_type}{seperator}'.format(
        decomposition_type=measurements_object.decomposition_type_correlations,
        seperator=measurements.universal.constants.SEPERATOR), '')
    file = file.replace('permutation_{permutation_method_decomposition_correlation}{seperator}'.format(
        permutation_method_decomposition_correlation=measurements_object.permutation_method_decomposition_correlation,
        seperator=measurements.universal.constants.SEPERATOR), '')
    # plot
    if use_abs:
        x_min = 0
        tick_number = 3
    else:
        x_min = -1
        tick_number = 5
    util.plot.histogram(data, file, step_size=0.1, x_min=x_min, x_max=1, tick_number=tick_number)

