import abc
import bisect

import numpy as np

import util.math.interpolate
import util.math.spherical

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



def periodic_with_coordinates(data, interpolation_points, lsm_base, scaling_values=None, interpolator_options=None):
    logger.debug('Interpolating periodic data with coordinates for lsm {} with scaling_values {} and interpolator_options {}.'.format(lsm_base, scaling_values, interpolator_options))
    
    ## convert coordinates to map indices
    data = np.array(data, copy=True)
    data[:, :-1] = lsm_base.coordinates_to_map_indices(data[:, :-1], discard_year=True, int_indices=False)
    interpolation_points = lsm_base.coordinates_to_map_indices(interpolation_points, discard_year=True, int_indices=False)

    ## interpolating
    return periodic_with_map_indices(data, interpolation_points, lsm_base, scaling_values=scaling_values, interpolator_options=interpolator_options)



def periodic_with_map_indices(data, interpolation_points, lsm_base, scaling_values=None, interpolator_options=None):
    logger.debug('Interpolating periodic data with map indices for lsm {} with scaling_values {} and interpolator_options {}.'.format(lsm_base, scaling_values, interpolator_options))
    
    assert data.ndim == 2
    assert data.shape[1] == 5
    assert interpolation_points.ndim == 2
    assert interpolation_points.shape[1] == 4

    ## split in points and values
    data_points = data[:, :-1]
    data_values = data[:, -1]

    ## scaling values
    if scaling_values is None:
        scaling_values=(lsm_base.x_dim/lsm_base.t_dim, None, None, None)
    
    if interpolator_options is None:
        interpolator_options = (1,1,0,0)

    ## prepare wrap_around_amount
    wrap_around_amount = interpolator_options[0]
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
    interpolator = util.math.interpolate.Periodic_Interpolator(data_points, data_values, point_range_size=(lsm_base.t_dim, lsm_base.x_dim, lsm_base.y_dim, lsm_base.z_dim), scaling_values=scaling_values, wrap_around_amount=wrap_around_amount, number_of_linear_interpolators=interpolator_options[1], single_overlapping_amount_linear_interpolators=interpolator_options[2], parallel=bool(interpolator_options[3]))

    ## interpolating
    interpolation_data = interpolator.interpolate(interpolation_points)
    return interpolation_data



def default_scaling_values(sample_lsm):
    scaling_x = 1
    scaling_y = sample_lsm.x_dim / (sample_lsm.y_dim * 2)
    if scaling_y.is_integer():
        scaling_y = int(scaling_y)
    scaling_t = sample_lsm.x_dim / (sample_lsm.t_dim * 3)
    if scaling_t.is_integer():
        scaling_t = int(scaling_t)
    scaling_z = int(np.floor(sample_lsm.x_dim / sample_lsm.z_dim))
    scaling_values = (scaling_t, scaling_x, scaling_y, scaling_z)
    return scaling_values
    
    

class Interpolator_Annual_Periodic:

    def __init__(self, sample_lsm, scaling_values=None):
        self.sample_lsm = sample_lsm
        self.scaling_values = scaling_values


    @property
    def scaling_values(self):
        return self._scaling_values

    @scaling_values.setter
    def scaling_values(self, scaling_values):
        if scaling_values is None:
            scaling_values = default_scaling_values(self.sample_lsm)
        self._scaling_values = scaling_values
    

    def interpolate_data_for_lsm(self, data, lsm, interpolator_options=None):
        logger.debug('Interpolating data for lsm {} with interpolator_options {}.'.format(lsm, interpolator_options))
        
        sea_indices = lsm.sea_indices
        sea_coordinates = lsm.map_indices_to_coordinates(sea_indices)
        
        interpolated_values = periodic_with_coordinates(data, sea_coordinates, self.sample_lsm, scaling_values=self.scaling_values, interpolator_options=interpolator_options)
        interpolated_data = np.concatenate((sea_indices, interpolated_values[:,np.newaxis]), axis=1)
        interpolated_map = lsm.insert_index_values_in_map(interpolated_data, no_data_value=np.inf)
        assert np.all(interpolated_map != np.inf)
        
        util.math.interpolate.change_dim(interpolated_map, 0, lsm.t_dim)
        assert interpolated_map.shape == lsm.dim
        
        return interpolated_map
    

    def interpolate_data_for_sample_lsm_with_coordinates(self, data, interpolator_options=None):
        logger.debug('Interpolating data with coordinates for lsm {} with interpolator_options {}.'.format(self.sample_lsm, interpolator_options))
        data = np.array(data, copy=True)
        data[:, :-1] = lsm_base.coordinates_to_map_indices(data[:, :-1], discard_year=True, int_indices=False)
        return self.interpolate_data_for_sample_lsm_with_map_indices(data, self.sample_lsm, interpolator_options=interpolator_options)


    def interpolate_data_for_sample_lsm_with_map_indices(self, data, interpolator_options=None):
        logger.debug('Interpolating data with map indices for lsm {} with interpolator_options {}.'.format(self.sample_lsm, interpolator_options))
        
        sea_indices = self.sample_lsm.sea_indices
        
        interpolated_values = periodic_with_map_indices(data, sea_indices, self.sample_lsm, scaling_values=self.scaling_values, interpolator_options=interpolator_options)
        interpolated_data = np.concatenate((sea_indices, interpolated_values[:,np.newaxis]), axis=1)
        
        interpolated_map = self.sample_lsm.insert_index_values_in_map(interpolated_data, no_data_value=np.inf)
        assert np.all(interpolated_map != np.inf)
        util.math.interpolate.change_dim(interpolated_map, 0, self.sample_lsm.t_dim)
        assert interpolated_map.shape == self.sample_lsm.dim
        
        return interpolated_map


    def interpolate_data_for_points_from_interpolated_lsm_data(self, interpolated_lsm_data, interpolation_points):
        logger.debug('Interpolationg data for points from interpolated data for lsm {}.'.format(self.sample_lsm))
        
        ## get interpolated points and values
        interpolated_lsm_data_mask = ~np.isnan(interpolated_lsm_data)
        interpolated_lsm_values = interpolated_lsm_data[interpolated_lsm_data_mask]
        interpolated_lsm_indices = np.array(np.where(interpolated_lsm_data_mask)).T
        interpolated_lsm_points = self.sample_lsm.map_indices_to_coordinates(interpolated_lsm_indices)
        interpolated_lsm_points_and_values = np.concatenate([interpolated_lsm_points, interpolated_lsm_values[:,np.newaxis]], axis=1)
        
        ## prepare interpolation points: convert to (rounded) int map indices -> convert to coordinates
        interpolation_points_map_indices = self.sample_lsm.coordinates_to_map_indices(interpolation_points, discard_year=True, int_indices=True)
        interpolation_points = self.sample_lsm.map_indices_to_coordinates(interpolation_points_map_indices)

        ## interpolate for points
        interpolated_points_data = periodic_with_coordinates(interpolated_lsm_points_and_values, interpolation_points, self.sample_lsm, scaling_values=self.scaling_values, interpolator_options=(2/min([self.sample_lsm.t_dim,self.sample_lsm.x_dim]),0,0,0))
        
        ## return
        assert np.all(np.isfinite(interpolated_points_data))
        return interpolated_points_data



    def interpolate_data_for_points(self, data, interpolation_points, interpolator_options=None):
        logger.debug('Interpolationg data for points with interpolator_options {}.'.format(interpolator_options))
        
        ## interpolate for sample lsm
        interpolated_lsm_data = self.interpolate_data_for_lsm(data, self.sample_lsm, interpolator_options=interpolator_options)
        
        ## interpolate for points
        interpolated_points_data = self.interpolate_data_for_points_from_interpolated_lsm_data(interpolated_lsm_data, interpolation_points)
        return interpolated_points_data
        
        # ## get interpolated points and values
        # interpolated_lsm_data_mask = ~np.isnan(interpolated_lsm_data)
        # interpolated_lsm_values = interpolated_lsm_data[interpolated_lsm_data_mask]
        # interpolated_lsm_indices = np.array(np.where(interpolated_lsm_data_mask)).T
        # interpolated_lsm_points = self.sample_lsm.map_indices_to_coordinates(interpolated_lsm_indices)
        # interpolated_lsm_points_and_values = np.concatenate([interpolated_lsm_points, interpolated_lsm_values[:,np.newaxis]], axis=1)
        # 
        # ## prepare interpolation points: convert to (rounded) int map indices -> convert to coordinates
        # interpolation_points_map_indices = self.sample_lsm.coordinates_to_map_indices(interpolation_points, discard_year=True, int_indices=True)
        # interpolation_points = self.sample_lsm.map_indices_to_coordinates(interpolation_points_map_indices)

        #   ## interpolate for points
        # interpolated_data = periodic_with_coordinates(interpolated_lsm_points_and_values, interpolation_points, self.sample_lsm, scaling_values=self.scaling_values, interpolator_options=(2/min([self.sample_lsm.t_dim,self.sample_lsm.x_dim]),0,0,0))
        # 
        # ## return
        # assert np.all(np.isfinite(interpolated_data))
        # return interpolated_data

