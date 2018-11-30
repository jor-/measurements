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


def per_time(measurements_object, step_size=None, file=None, overwrite=False):
    if step_size is None:
        step_size = 1
    else:
        step_size = _get_int_if_possible(step_size)

    if file is None:
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='number_of_measurements',
            kind_id=str(measurements_object.sample_lsm),
            plot_name=f'number_of_measurements_per_time_-_step_size_{step_size}')

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
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='number_of_measurements',
            kind_id=str(measurements_object.sample_lsm),
            plot_name=f'number_of_measurements_within_a_year_-_number_of_bins_{number_of_bins}')

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
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='number_of_measurements',
            kind_id=str(measurements_object.sample_lsm),
            plot_name=plot_name)

    points = measurements_object.points
    assert points.ndim == 2
    assert points.shape[1] == 4
    z = points[:, 3]

    util.plot.save.histogram(file, z, step_size=step_size, use_log_scale=use_log_scale,
                             tick_power_limit_scientific_y=tick_power_limit_scientific_y, overwrite=overwrite)


def per_space(measurements_object, use_log_scale=True, file=None, overwrite=False):
    if file is None:
        plot_name = 'number_of_measurements_per_space'
        if use_log_scale:
            plot_name += '_-_log_scale'
            tick_power_limit_scientific_y = None
        else:
            tick_power_limit_scientific_y = 3
        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=measurements_object.tracer,
            data_set=measurements_object.data_set_name,
            kind='number_of_measurements',
            kind_id=str(measurements_object.sample_lsm),
            plot_name=plot_name)

    points = measurements_object.points
    assert points.ndim == 2
    assert points.shape[1] == 4
    space_points = points[:, 1:]
    lsm = measurements_object.sample_lsm
    space_coordinates = lsm.coordinates_to_map_indices(space_points, int_indices=True)
    data, counts = np.unique(space_coordinates, axis=0, return_counts=True)
    data = np.concatenate((data, counts[:, np.newaxis]), axis=1)
    data = lsm.insert_index_values_in_map(data, no_data_value=np.inf)

    util.plot.save.data(file, data, no_data_value=np.inf,
                        use_log_scale=use_log_scale, contours=False, colorbar=True,
                        tick_power_limit_scientific_y=tick_power_limit_scientific_y, overwrite=overwrite)
