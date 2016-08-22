import bisect

import numpy as np

import util.math.spherical
import util.math.interpolate

import util.logging
logger = util.logging.logger



class Time_Periodic_Earth_Interpolator(util.math.interpolate.Periodic_Interpolator):

    def __init__(self, data_points, data_values, t_len, wrap_around_amount=0, number_of_linear_interpolators=1, single_overlapping_amount_linear_interpolators=0, parallel=False):
        from measurements.constants import EARTH_RADIUS

        logger.debug('Initiating time periodic earth interpolator with {} data points, time len {}, wrap around amount {} and {} linear interpolators with single overlapping amount of {}.'.format(len(data_points), t_len, wrap_around_amount, number_of_linear_interpolators, single_overlapping_amount_linear_interpolators))

        ## call super constructor
        self.order = number_of_linear_interpolators

        t_scaling = 2 * EARTH_RADIUS / t_len

        super().__init__(data_points, data_values, point_range_size=(t_len, None, None, None), wrap_around_amount=(wrap_around_amount, 0, 0, 0),  scaling_values=(t_scaling, None, None, None), number_of_linear_interpolators=number_of_linear_interpolators, single_overlapping_amount_linear_interpolators=single_overlapping_amount_linear_interpolators, parallel=parallel)

        assert len(self._data_points) == len(self._data_values) == len(self._data_indices)


    def _modify_points(self, points, is_data_points):
        from measurements.constants import EARTH_RADIUS, MAX_SEA_DEPTH

        points = super()._modify_points(points, is_data_points)

        ## if data points, append values for lower and upper bound of depth
        if self.order > 0 and is_data_points:
            lower_depth = 0
            lower_depth_bound = np.min(points[:,3])
            upper_depth = MAX_SEA_DEPTH
            upper_depth_bound = np.max(points[:,3])

            logger.debug('Lower depth is {}, upper depth is {}.'.format(lower_depth, upper_depth))

            assert lower_depth_bound >= lower_depth and upper_depth_bound <= upper_depth

            if lower_depth_bound > lower_depth:
                lower_depth_bound_indices = np.where(np.isclose(points[:,3], lower_depth_bound))[0]
                lower_depth_bound_points = points[lower_depth_bound_indices]
                lower_depth_bound_points[:,3] = lower_depth
                logger.debug('{} values appended for lower bound {}.'.format(len(lower_depth_bound_indices), lower_depth))
            else:
                lower_depth_bound_indices = np.array([])
                lower_depth_bound_points = np.array([])
                logger.debug('No values appended for lower bound {}.'.format(lower_depth))
            if upper_depth_bound < upper_depth:
                upper_depth_bound_indices = np.where(np.isclose(points[:,3], lower_depth_bound))[0]
                upper_depth_bound_points = points[upper_depth_bound_indices]
                upper_depth_bound_points[:,3] = upper_depth
                logger.debug('{} values appended for upper bound {}.'.format(len(upper_depth_bound_indices), upper_depth))
            else:
                upper_depth_bound_indices= np.array([])
                upper_depth_bound_points = np.array([])
                logger.debug('No values appended for upper bound {}.'.format(upper_depth))

            indices = np.concatenate((lower_depth_bound_indices, np.arange(len(points)), upper_depth_bound_indices), axis=0)
            points = np.concatenate((lower_depth_bound_points, points, upper_depth_bound_points), axis=0)
            self._data_indices = self._data_indices[indices]

        ## convert to cartesian
        points[:,1:] =  util.math.spherical.to_cartesian(points[:,1:], surface_radius=EARTH_RADIUS)

        return points



## interpolate

def periodic_with_coordinates(data, interpolation_points, lsm_base, scaling_values=None, interpolator_setup=(0.1,1,0,0)):
    assert data.ndim == 2
    assert data.shape[1] == 5
    assert interpolation_points.ndim == 2
    assert interpolation_points.shape[1] == 4

    ## split in points and values
    data_points = data[:, :-1]
    data_values = data[:, -1]

    ## convert point to box indices
    data_points = lsm_base.coordinates_to_map_indices(data_points, discard_year=True, int_indices=False)
    interpolation_points = lsm_base.coordinates_to_map_indices(interpolation_points, discard_year=True, int_indices=False)

    ## scaling values
    if scaling_values is None:
        scaling_values=(lsm_base.x_dim/lsm_base.t_dim, None, None, None)

    ## prepare wrap_around_amount
    wrap_around_amount=interpolator_setup[0]
    try:
        wrap_around_amount = tuple(wrap_around_amount)
    except TypeError:
        wrap_around_amount = (wrap_around_amount,)
    if len(wrap_around_amount) == 1:
        ## use same wrap around for t and x
        wrap_around_amount = wrap_around_amount * 2
    if len(wrap_around_amount) == 2:
        ## append wrap around for y and z if missing
        wrap_around_amount = wrap_around_amount + (0,0)

    ## create interpolator
    interpolator = util.math.interpolate.Periodic_Interpolator(data_points, data_values, point_range_size=(lsm_base.t_dim, lsm_base.x_dim, lsm_base.y_dim, lsm_base.z_dim), scaling_values=scaling_values, wrap_around_amount=wrap_around_amount, number_of_linear_interpolators=interpolator_setup[1], single_overlapping_amount_linear_interpolators=interpolator_setup[2], parallel=bool(interpolator_setup[3]))

    ## interpolating
    interpolation_data = interpolator.interpolate(interpolation_points)
    return interpolation_data

