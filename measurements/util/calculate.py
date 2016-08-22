import numpy as np


def average_over_time(data):
    averaged_data = np.nansum(data, axis=0) / np.sum(np.logical_not(np.isnan(data)), axis=0)
    return averaged_data


def wrap_around_index(index, index_range):
    if index_range is not None:
        index_range_diff = index_range[1] - index_range[0]
        while index < index_range[0]:
            index += index_range_diff
        while index >= index_range[1]:
            index -= index_range_diff
    return index


def normalize_points(points, ranges):
    points = np.asanyarray(points)
    ranges = np.asanyarray(ranges)
    return (points - range[:,0]) / (range[:,1] - range[:,0])


def get_min_distance(point_1, point_2, t_range=None, x_range=None):
    distance = np.abs(point_1 - point_2)

    ## wrap around
    if t_range is not None:
        t_size = t_range[1] - t_range[0]
        if distance[0] > t_size / 2:
            distance[0] -= t_size / 2

    if x_range is not None:
        x_size = x_range[1] - x_range[0]
        if distance[1] > x_size / 2:
            distance[1] -= x_size / 2

    return distance

