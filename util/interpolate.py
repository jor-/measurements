import numpy as np
import logging

import util.spherical
import util.interpolate






class Time_Periodic_Interpolator(util.interpolate.Interpolater):
    
    def __init__(self, data_points, data_values, t_len, wrap_around_amount=0, number_of_linear_interpolators=1, total_overlapping_linear_interpolators=0, t_scaling=1, copy_arrays=True, parallel=False):
        
        logging.debug('Initiating time periodic interpolator with {} data points, time len {}, wrap around amount {}, {} linear interpolators with total overlapping of {}.'.format(len(data_points), t_len, wrap_around_amount, number_of_linear_interpolators, total_overlapping_linear_interpolators))
        
#         #assert np.max(data_points[0,:]) - np.min(data_points[0,:]) <= t_len
#         
#         ## discard year
#         data_points = np.array(data_points, copy=True)
#         data_points[:,0] = data_points[:,0] % t_len
#         self._t_len = t_len
#         
#         ## wrap around t
#         data_points, data_indices = util.interpolate.wrap_around(data_points, 0, t_len, amount=wrap_around_amount, return_also_indices=True)
#         self._data_indices = data_indices
        self._t_len = t_len
        self._wrap_around_amount = wrap_around_amount
        
        super().__init__(data_points, data_values, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, scaling_values=(t_scaling, 1, 1, 1), copy_arrays=copy_arrays, parallel=parallel)
        
        assert len(self._data_points) == len(self._data_values) == len(self._data_indices)
    
    
    def _prepare_data_values(self, data_values):
        logging.debug('Wrapping around data values.')
        
        data_values = data_values[self._data_indices]
        data_values = super()._prepare_data_values(data_values)
        
        return data_values
    
    
    def _modify_points(self, points, is_data_points):
        ## discard time for periodicity
        t_len = self._t_len
        logging.debug('Discarding time of points which is a multiple of {}.'.format(t_len))
        points[:,0] = points[:,0] % t_len
        
        ## if data points, wrap around t
        if is_data_points:
            points, indices = util.interpolate.wrap_around(points, 0, t_len, amount=self._wrap_around_amount, return_also_indices=True)
            self._data_indices = indices
        
        return points
    
    
    def _modify_interpolation_points(self, points):
        points = self._modify_points(points, False)
        points = super()._modify_interpolation_points(points)
        return points
    
    
    def _modify_data_points(self, points):
        points = self._modify_points(points, True)
        points = super()._modify_data_points(points)
        return points
    
    
    






class Time_Periodic_Non_Cartesian_Interpolator(Time_Periodic_Interpolator):
    
    def __init__(self, data_points, data_values, t_len, x_len, t_scale=True, wrap_around_amount=0, number_of_linear_interpolators=1, total_overlapping_linear_interpolators=0, parallel=False):
        
        logging.debug('Initiating time periodic non spherical interpolator with {} data points, time len {}, x len {}, time scale {}, wrap around amount {} and {} linear interpolators with total overlapping of {}.'.format(len(data_points), t_len, x_len, t_scale, wrap_around_amount, number_of_linear_interpolators, total_overlapping_linear_interpolators))
        
        assert np.max(data_points[:,1]) - np.min(data_points[:,1]) <= x_len
        
#         ## wrap around x
#         data_points, data_indices = util.interpolate.wrap_around(data_points, 1, x_len, amount=wrap_around_amount, return_also_indices=True)
#         data_values = data_values[data_indices]
#         
#         ## call super constructor
        
        self._x_len = x_len
        if t_scale:
            t_scaling = x_len / t_len
        else:
            t_scaling = 1
        
        super().__init__(data_points, data_values, t_len, wrap_around_amount=wrap_around_amount, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, t_scaling=t_scaling, parallel=parallel)
        
#         ## set indices with x wrap around
#         self._data_indices = data_indices[self._data_indices]
        
        assert len(self._data_points) == len(self._data_values) == len(self._data_indices)
    
    
    
    def _modify_points(self, points, is_data_points):
        points = super()._modify_points(points, is_data_points)
        
        ## if data points, wrap around x
        x_len = self._x_len
        if is_data_points:
            points, indices = util.interpolate.wrap_around(points, 1, x_len, amount=self._wrap_around_amount, return_also_indices=True)
            self._data_indices = self._data_indices[indices]
        
        return points
    




class Time_Periodic_Earth_Interpolater(Time_Periodic_Interpolator):
    
    
    def __init__(self, data_points, data_values, t_len, wrap_around_amount=0, number_of_linear_interpolators=1, total_overlapping_linear_interpolators=0, parallel=False):
        from measurements.constants import EARTH_RADIUS
        
        logging.debug('Initiating time periodic earth interpolator with {} data points, time len {}, wrap around amount {} and {} linear interpolators with total overlapping of {}.'.format(len(data_points), t_len, wrap_around_amount, number_of_linear_interpolators, total_overlapping_linear_interpolators))
        
        
#         ## if data points, append values for lower and upper bound of depth
#         lower_depth_bound = np.min(data_points[3,:])
#         upper_depth_bound = np.max(data_points[3,:])
#         
#         assert lower_depth_bound >= 0 and upper_depth_bound <= MAX_SEA_DEPTH
#         
#         if lower_depth_bound > 0:
#             lower_depth_bound_data_indices = np.where(data_points[3,:] == lower_depth_bound)[0]
#             lower_depth_bound_data_points = data_points[lower_depth_bound_data_indices]
#             lower_depth_bound_data_points[3,:] = 0
#         else:
#             lower_depth_bound_data_mask = np.array([])
#         if upper_depth_bound < MAX_SEA_DEPTH:
#             upper_depth_bound_data_indices = np.where(data_points[3,:] == lower_depth_bound)[0]
#             upper_depth_bound_data_points = data_points[upper_depth_bound_data_indices]
#             upper_depth_bound_data_points[3,:] = MAX_SEA_DEPTH
#         
#         data_points = np.concatenate((lower_depth_bound_data_points, data_points, upper_depth_bound_data_points), axis=0)
#         data_indices = np.concatenate((lower_depth_bound_data_indices, np.arange(len(data_values)), upper_depth_bound_data_indices), axis=0)
#         
#         data_values = data_values[data_indices]
#         
#         ## call super constructor
        self.order = number_of_linear_interpolators
        
        t_scaling = 4 * EARTH_RADIUS / t_len
        
        super().__init__(data_points, data_values, t_len, wrap_around_amount=wrap_around_amount, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, t_scaling=t_scaling, parallel=parallel)
        
#         ## set indices with depth shift
#         self._data_indices = data_indices[self._data_indices]
        
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
            
            logging.debug('Lower depth is {}, upper depth is {}.'.format(lower_depth, upper_depth))
            
            assert lower_depth_bound >= 0 and upper_depth_bound <= upper_depth
            
            if lower_depth_bound > lower_depth:
                lower_depth_bound_indices = np.where(points[:,3] == lower_depth_bound)[0]
                lower_depth_bound_points = points[lower_depth_bound_indices]
                lower_depth_bound_points[:,3] = lower_depth
                logging.debug('{} values appended for lower bound {}.'.format(len(lower_depth_bound_indices), lower_depth))
            else:
                lower_depth_bound_indices = np.array([])
                lower_depth_bound_points = np.array([])
                logging.debug('No values appended for lower bound {}.'.format(lower_depth))
            if upper_depth_bound < upper_depth:
                upper_depth_bound_indices = np.where(points[:,3] == lower_depth_bound)[0]
                upper_depth_bound_points = points[upper_depth_bound_indices]
                upper_depth_bound_points[:,3] = upper_depth
                logging.debug('{} values appended for upper bound {}.'.format(len(upper_depth_bound_indices), upper_depth))
            else:
                upper_depth_bound_indices= np.array([])
                upper_depth_bound_points = np.array([])
                logging.debug('No values appended for upper bound {}.'.format(upper_depth))
            
            indices = np.concatenate((lower_depth_bound_indices, np.arange(len(points)), upper_depth_bound_indices), axis=0)
            points = np.concatenate((lower_depth_bound_points, points, upper_depth_bound_points), axis=0)
            self._data_indices = self._data_indices[indices]
        
        ## convert to cartesian
        points[:,1:] =  util.spherical.to_cartesian(points[:,1:], surface_radius=EARTH_RADIUS)
        
        return points
 
     