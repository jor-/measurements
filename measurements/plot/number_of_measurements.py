import os.path

import numpy as np

import util.plot.save
import measurements.plot.constants


def _get_int_if_possible(value):
    try:
        int_value = int(value)
    except ValueError:
        return value
    else:
        if int_value == value:
            return int_value
        else:
            return value


def _format_filename(measurements_object, plot_name):
    file = measurements.plot.constants.PLOT_FILE.format(
        tracer=measurements_object.tracer,
        data_set=measurements_object.data_set_name,
        kind='number_of_measurements',
        kind_id=str(measurements_object.sample_lsm),
        plot_name=plot_name,
        file_extension=measurements.plot.constants.PLOT_DEFAULT_FILE_EXTENSION)
    return file


def per_time(measurements_object, step_size=None, file=None, overwrite=False):
    if step_size is None:
        step_size = 1
    else:
        step_size = _get_int_if_possible(step_size)

    if file is None:
        plot_name = f'number_of_measurements_per_time_-_step_size_{step_size}'
        file = _format_filename(measurements_object, plot_name)

    points = measurements_object.points
    assert points.ndim == 2
    assert points.shape[1] == 4
    t = points[:, 0]

    util.plot.save.histogram(file, t, step_size=step_size,
                             tick_power_limit_scientific_y=3, overwrite=overwrite)


def per_year(measurements_object, number_of_bins=None, file=None, overwrite=False):
    if number_of_bins is None:
        number_of_bins = 365
    else:
        assert int(number_of_bins) == number_of_bins
        number_of_bins = int(number_of_bins)
    step_size = 1. / number_of_bins

    if file is None:
        plot_name = f'number_of_measurements_within_a_year_-_number_of_bins_{number_of_bins}'
        file = _format_filename(measurements_object, plot_name)

    points = measurements_object.points
    assert points.ndim == 2
    assert points.shape[1] == 4
    t = points[:, 0]
    t = t % 1

    util.plot.save.histogram(file, t, step_size=step_size,
                             tick_power_limit_scientific_y=3, overwrite=overwrite)


def per_depth(measurements_object, step_size=None, use_log_scale=True, file=None, overwrite=False):
    if step_size is None:
        step_size = 50
    else:
        step_size = _get_int_if_possible(step_size)

    if file is None:
        plot_name = f'number_of_measurements_per_depth_-_step_size_{step_size}'
        if use_log_scale:
            plot_name += '_-_log_scale'
            tick_power_limit_scientific_y = None
        else:
            tick_power_limit_scientific_y = 3
        file = _format_filename(measurements_object, plot_name)

    points = measurements_object.points
    assert points.ndim == 2
    assert points.shape[1] == 4
    z = points[:, 3]

    util.plot.save.histogram(file, z, step_size=step_size, use_log_scale=use_log_scale,
                             tick_power_limit_scientific_y=tick_power_limit_scientific_y, overwrite=overwrite)


def _number_of_measurements_map(measurements_object, no_data_value=0, land_value=np.nan):
    # calculate number of measurements including layers
    points = measurements_object.points
    assert points.ndim == 2
    assert points.shape[1] == 4
    space_points = points[:, 1:]
    lsm = measurements_object.sample_lsm
    space_coordinates = lsm.coordinates_to_map_indices(space_points, int_indices=True)
    data, counts = np.unique(space_coordinates, axis=0, return_counts=True)
    data = np.concatenate((data, counts[:, np.newaxis]), axis=1)
    number_of_measurements_map = lsm.insert_index_values_in_map(data, no_data_value=no_data_value, land_value=land_value)
    assert number_of_measurements_map.ndim == 3
    return number_of_measurements_map


def per_space_each_depth(measurements_object, max_value_fixed=True, use_log_scale=True, file=None, overwrite=False):
    # prepare filename
    if file is None:
        plot_name = 'number_of_measurements_per_space_depth_{depth}'
        if use_log_scale:
            plot_name += '_-_log_scale'
            tick_power_limit_scientific_y = None
        else:
            tick_power_limit_scientific_y = 3
        if max_value_fixed:
            plot_name += '_-_max_value_fixed'
        file = _format_filename(measurements_object, plot_name)
    # calculate number of measurements including layers
    no_data_value = 0
    land_value = np.nan
    data = _number_of_measurements_map(measurements_object, no_data_value=no_data_value, land_value=land_value)
    if max_value_fixed:
        data_mask = np.logical_and(np.logical_not(np.isnan(data)), np.logical_not(data == np.inf))
        v_max = np.max(data[data_mask])
    else:
        v_max = None
    # plot number of measurements for each layers
    util.plot.save.data(file, data, no_data_value=no_data_value, land_value=land_value, v_min=1, v_max=v_max,
                        use_log_scale=use_log_scale, contours=False, colorbar=True,
                        tick_power_limit_scientific_y=tick_power_limit_scientific_y, overwrite=overwrite)


def per_space(measurements_object, use_log_scale=True, file=None, overwrite=False):
    # prepare filename
    if file is None:
        plot_name = 'number_of_measurements_per_space'
        if use_log_scale:
            plot_name += '_-_log_scale'
            tick_power_limit_scientific_y = None
        else:
            tick_power_limit_scientific_y = 3
        file = _format_filename(measurements_object, plot_name)
    # calculate number of measurements add up all layers
    no_data_value = 0
    land_value = np.nan
    data = _number_of_measurements_map(measurements_object, no_data_value=no_data_value, land_value=land_value)
    land_mask = np.isnan(data[:, :, 0])
    data = np.nansum(data, axis=2)
    data[land_mask] = land_value
    # plot summed number of measurements
    util.plot.save.data(file, data, no_data_value=no_data_value, land_value=land_value, v_min=1, v_max=None,
                        use_log_scale=use_log_scale, contours=False, colorbar=True,
                        tick_power_limit_scientific_y=tick_power_limit_scientific_y, overwrite=overwrite)


def _number_of_estimations(measurements_object, min_number_of_measurements_for_estimation=1, min_number_of_estimations=1):
    points = measurements_object.points
    lsm = m.sample_lsm
    map_indices = m.sample_lsm.coordinates_to_map_indices(points, int_indices=True, discard_year=False)
    data, counts = np.unique(map_indices, axis=0, return_counts=True)
    data = data[counts >= min_number_of_measurements_for_estimation]
    data[:, 0] = data[:, 0] % lsm.t_dim
    data, counts = np.unique(map_indices, axis=0, return_counts=True)
    data_with_counts = np.concatenate(data, counts, axis=1)
    data_with_counts = data_with_counts[counts >= min_number_of_estimations]
    return data_with_counts
