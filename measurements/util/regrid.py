import bisect
import logging

import numpy as np

import util.math.interpolate


## convert point to box index (helper functions)

def get_spatial_float_index(x, y, z, land_sea_mask, z_values_left, x_range=[0, 360], y_range=[-90, 90]):
    logging.debug('Getting spatial float index for {}.'.format((x, y, z)))

    ## adjust x coordinates if negative
    if x < 0:
        x += 360

    ## check input
    if x < x_range[0] or x > x_range[1]:
        raise ValueError('Value {} of x is not in range {}.'.format(x, x_range))
    if y < y_range[0] or y > y_range[1]:
        raise ValueError('Value {} of y is not in range {}.'.format(y, y_range))
    if z < z_values_left[0]:
        raise ValueError('Value {} of z have to be greater or equal to {}.'.format(z, z_values_left[0]))

    ## linear interpolate x and y index
    (x_dim, y_dim) = land_sea_mask.shape
    x_index_float = util.math.interpolate.get_float_index_for_equidistant_values(x, x_range, x_dim)
    y_index_float = util.math.interpolate.get_float_index_for_equidistant_values(y, y_range, y_dim)

    ## lockup z
    z_index = bisect.bisect_right(z_values_left, z) - 1

    if z_index + 1 < len(z_values_left):
        z_left = z_values_left[z_index]
        z_right = z_values_left[z_index + 1]
        z_index_float = z_index + (z - z_left) / (z_right - z_left)
    else:
        z_index_float = z_index


    logging.debug('Float indices for {} are {}.'.format((x, y, z), (x_index_float, y_index_float, z_index_float)))

    return (x_index_float, y_index_float, z_index_float)


def get_temporal_float_index(t, t_dim, t_range=[0, 1]):
    logging.debug('Getting temporal float index for {}.'.format(t))

    ## check input
    if t < t_range[0] or t > t_range[1]:
        raise ValueError('Value {} of t is not in range {}.'.format(t, t_range))

    ## interpolate
    t_index_float = util.math.interpolate.get_float_index_for_equidistant_values(t, t_range, t_dim)

    logging.debug('Temporal float index for {} is {}.'.format(t, t_index_float))

    return t_index_float


def is_water(spatial_indices, land_sea_mask):
    logging.debug('Checking if indices are water.')

    ## check input
    spatial_indices = np.array(spatial_indices, dtype=np.int)

    dim = len(spatial_indices.shape)

    if dim > 2 or dim == 0:
        raise ValueError('The spatial_indices array has to have 1 or 2 dimensions, but it has {} dimensions.'.format(dim))
    elif dim == 1:
        spatial_len = spatial_indices.shape[0]
        if spatial_len != 3:
            raise ValueError('The len of spatial_indices array has to be 3, but it is {}.'.format(spatial_len))
        spatial_indices = spatial_indices.reshape([1, spatial_len])
    else:
        spatial_len = spatial_indices.shape[1]
        if spatial_len != 3:
            raise ValueError('The second dimension of the spatial_indices array to be 3, but it is {}.'.format(spatial_len))

    ## compute if water
    land_sea_mask_indices = spatial_indices[:, :-1]
    land_sea_mask_values = land_sea_mask[land_sea_mask_indices[:, 0], land_sea_mask_indices[:, 1]]
    is_water = np.logical_or(np.isnan(land_sea_mask_values),  land_sea_mask_values >= spatial_indices[:, -1])

    return is_water


def get_all_water_boxes(land_sea_mask):
    logging.debug('Getting all water boxes.')

    land_sea_mask = land_sea_mask.lsm
    (water_x, water_y) = np.where(land_sea_mask != 0)
    water_len = np.sum(land_sea_mask)
    water_boxes = np.empty([water_len, 3], dtype=np.int)

    j = 0
    for i in range(len(water_x)):
        x_i = water_x[i]
        y_i = water_y[i]
        z_i = land_sea_mask[x_i, y_i]

        j_range = range(j, j + z_i)
        water_boxes[j_range, 0] = x_i
        water_boxes[j_range, 1] = y_i
        water_boxes[j_range, 2] = range(z_i)

        j += z_i

    return water_boxes


def get_nearest_water_box(land_sea_mask, x_index, y_index, z_index):
    logging.debug('Getting nearest water box for index {}.'.format((x_index, y_index, z_index)))

    water_boxes = get_all_water_boxes(land_sea_mask)

    index = [x_index, y_index, z_index]
    nearest_water_box = util.math.interpolate.get_nearest_value_in_array(water_boxes, index)

    assert land_sea_mask[nearest_water_box[0], nearest_water_box[1]] >= nearest_water_box[2]

    return nearest_water_box


def get_nearest_spatial_water_index(x, y, z, land_sea_mask, z_values_left, x_range=[0, 360], y_range=[-90, 90]):
    (x_index_float, y_index_float, z_index_float) = get_spatial_float_index(x, y, z, land_sea_mask, z_values_left, x_range=x_range, y_range=y_range)

    ## floor indices to int
    x_index = np.floor(x_index_float)
    y_index = np.floor(y_index_float)
    z_index = np.floor(z_index_float)

    ## get nearest water box if box is land
    if not is_water((x_index, y_index, z_index), land_sea_mask):
        logging.debug('Box {} is land.'.format((x_index, y_index, z_index)))
        (x_index, y_index, z_index) = get_nearest_water_box(land_sea_mask, x_index_float, y_index_float, z_index_float)

    logging.debug('Nearest index for {} is {}.'.format((x, y, z), (x_index_float, y_index_float, z_index_float)))

    assert land_sea_mask[x_index, y_index] >= z_index

    return (x_index, y_index, z_index)


def get_nearest_temporal_index(t, t_dim, t_range=[0, 1]):
    t_index_float = get_temporal_float_index(t, t_dim, t_range=t_range)
    t_index = np.floor(t_index_float)

    logging.debug('Nearest temporal index for {} is {}.'.format(t, t_index))

    return t_index


def get_nearest_water_index(t, x, y, z, land_sea_mask, z_values_left, t_range=[0, 1], x_range=[0, 360], y_range=[-90, 90]):
    t_index =  get_nearest_temporal_index(t, land_sea_mask.t_dim, t_range=t_range)
    x_index, y_index, z_index = get_nearest_spatial_water_index(x, y, z, land_sea_mask, z_values_left, x_range=x_range, y_range=y_range)

    assert land_sea_mask[x_index, y_index] >= z_index

    return (t_index, x_index, y_index, z_index)


def convert_point_to_box_index(t, x, y, z, land_sea_mask, z_values_left, t_range=[0, 1], x_range=[0, 360], y_range=[-90, 90]):
    return get_nearest_water_index(t, x, y, z, land_sea_mask, z_values_left, t_range=t_range, x_range=x_range, y_range=y_range)




## regird

def measurements_to_land_sea_mask(measurement_data, land_sea_mask, t_range=[0, 1], x_range=[0, 360], y_range=[-90, 90]):
    assert measurement_data.ndim == 2
    assert measurement_data.shape[1] in [5, 7]

    z_values_left = land_sea_mask.z_left

    ## init arrays
    nobs = land_sea_mask.masked_map(default_value=0, dtype=np.float64)
    sum_of_values = np.copy(nobs)
    sum_of_squares = np.copy(nobs)
    variances = land_sea_mask.masked_map(default_value=np.inf, dtype=np.float64)

    ## init dim values
    number_of_measurements = measurement_data.shape[0]
    data_dim = measurement_data.shape[1]

    ## insert measurements
    for i in range(number_of_measurements):
        t, x, y, z, data_sum_of_values = measurement_data[i, :5]
        if data_dim == 5:
            data_nobs = 1
            data_sum_of_squares = data_sum_of_values**2
        else:
            data_nobs, data_sum_of_squares = measurement_data[i, 5:]

        (t_index, x_index, y_index, z_index) = convert_point_to_box_index(t, x, y, z, land_sea_mask, z_values_left, t_range=t_range, x_range=x_range, y_range=y_range)

        nobs[t_index, x_index, y_index, z_index] += data_nobs
        sum_of_values[t_index, x_index, y_index, z_index] += data_sum_of_values
        sum_of_squares[t_index, x_index, y_index, z_index] += data_sum_of_squares

    ## calculate mean
    where_measurements = np.where(nobs > 0)

    means = np.copy(sum_of_values)
    means[where_measurements] /= nobs[where_measurements]

    ## calculate variance
    where_measurements_over_threshold = np.where(nobs >= 2)

    nobs_over_threshold = nobs[where_measurements_over_threshold]
    sum_of_values_over_threshold = sum_of_values[where_measurements_over_threshold]
    sum_of_squares_over_threshold = sum_of_squares[where_measurements_over_threshold]

    variances[where_measurements_over_threshold] = (sum_of_squares_over_threshold - sum_of_values_over_threshold ** 2 / nobs_over_threshold) / (nobs_over_threshold - 1)

    return (means, nobs, variances)
