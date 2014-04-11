import numpy as np
import logging

from . import estimation


import measurements.util.interpolate




class Deviation_Model():
    
    def __init__(self, measurements_file, separation_values, t_len, x_len, minimum_measurements=5, convert_spherical_to_cartesian=True, wrap_around_amount=1, number_of_linear_interpolators=1, total_overlapping_linear_interpolators=0, parallel=False):
        
        ## estimate deviation
        deviation = estimation.deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file)
        
        ## split deviation in points and values
        deviation_points = deviation[:, :-1]
        deviation_values = deviation[:, -1]
        
        if convert_spherical_to_cartesian:
            self.interpolator = measurements.util.interpolate.Time_Periodic_Earth_Interpolater(deviation_points, deviation_values, t_len, wrap_around_amount=wrap_around_amount, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, parallel=parallel)
        else:
            self.interpolator = measurements.util.interpolate.Time_Periodic_Non_Cartesian_Interpolator(deviation_points, deviation_values, t_len, x_len, wrap_around_amount=wrap_around_amount, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, parallel=parallel)
    
    
    
    def deviation(self, interpolation_points):
        logging.debug('Interpolating standard deviation for {} points.'.format(interpolation_points.shape[0]))
        
        ## interpolating
        deviation = self.interpolator.interpolate(interpolation_points)
        
        return deviation
    
    









# import util.spherical
# import util.interpolate
# 
# 
# 
# 
# class Deviation_Model():
#     
#     def __init__(self, measurements_file, separation_values, t_len, x_len, minimum_measurements=5, convert_spherical_to_cartesian=True, wrap_around_amount=1):
#         from measurements.constants import EARTH_RADIUS, MAX_SEA_DEPTH
#         
#         self.EARTH_RADIUS = EARTH_RADIUS
#         self.convert_spherical_to_cartesian = convert_spherical_to_cartesian
#         
#         #TODO  move in seperate "EARTH" interpolator
#         
#         ## estimate deviation
#         deviation = estimation.deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file)
#         
#         logging.debug('Got standard deviation estimation at %d points.' % deviation.shape[0])
#         
#         assert np.max(deviation[0,:]) - np.min(deviation[0,:]) <= t_len
#         assert np.max(deviation[1,:]) - np.min(deviation[1,:]) <= x_len
#         
#         
#         ## append values for lower and upper bound of depth
#         lower_depth_bound = np.min(deviation[3,:])
#         upper_depth_bound = np.max(deviation[3,:])
#         
#         assert lower_depth_bound >= 0 and upper_depth_bound <= MAX_SEA_DEPTH
#         
#         if lower_depth_bound > 0:
#             lower_depth_bound_values = deviation[deviation[3,:] == lower_depth_bound]
#             lower_depth_bound_values[3,:] = 0
#         if upper_depth_bound < MAX_SEA_DEPTH:
#             upper_depth_bound_values = deviation[deviation[3,:] == upper_depth_bound]
#             upper_depth_bound_values[3,:] = MAX_SEA_DEPTH
#         
#         deviation = np.concatenate((lower_depth_bound_values, deviation, upper_depth_bound_values), axis=0)
#         
#         
#         
#         ## wrap around
#         deviation = util.interpolate.wrap_around(deviation, 0, t_len, amount=wrap_around_amount)
#         if not convert_spherical_to_cartesian:
#             deviation = util.interpolate.wrap_around(deviation, 1, x_len, amount=wrap_around_amount)
#         
#         
#         ## calculate t scaling factor
#         if convert_spherical_to_cartesian:
#             t_scaling_factor = 4 * EARTH_RADIUS / t_len
#         else:
#             t_scaling_factor = x_len / t_len
#         self.t_scaling_factor = t_scaling_factor
#         
#         
# #         ## sort deviation by t
# #         deviation = deviation[np.argsort(deviation[:,0])]
#         
#         
#         ## split deviation in points and values
#         deviation_points = deviation[:, :-1]
#         deviation_values = deviation[:, -1]
#         
#         self.deviation_points = self.convert_points(deviation_points)
#         self.deviation_values = deviation_values
#     
#     
#     def convert_points(self, points):
# #         t_scaling_factor = self.t_scaling_factor
# #         
# #         logging.debug('Scaling t by factor {}.'.format(t_scaling_factor))
# #         points[:,0] *= t_scaling_factor
#         
#         if self.convert_spherical_to_cartesian:
#             points[:,1:] = util.spherical.to_cartesian(points[:,1:], surface_radius=self.EARTH_RADIUS)
#         
#         return points
#     
#     
#     def deviation(self, interpolation_points, number_of_linear_interpolators=1, total_overlapping_linear_interpolators=0):
#         logging.debug('Interpolating standard deviation for {} points.'.format(interpolation_points.shape[0]))
#         
#         ## prepare interpolation points
#         # copy points
#         interpolation_points = np.array(interpolation_points, copy=True)
#         
#         # make 2 dim
#         if interpolation_points.ndim == 1:
#             interpolation_points = interpolation_points[None, :]
#         
#         # discard year
#         interpolation_points[:, 0] = interpolation_points[:, 0] % 1
#         
#         # convert to cartesian
#         interpolation_points = self.convert_points(interpolation_points)
#         
#         ## interpolating
#         deviation = util.interpolate.interpolate(self.deviation_points, self.deviation_values, interpolation_points, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, scaling_values=[self.t_scaling_factor, 1, 1, 1])
#         
#         return deviation
#     
#     
# 



# 
# 
# class Deviation_Model():
#     
#     def __init__(self, measurements_file, separation_values, t_range, x_range, minimum_measurements=5, convert_spherical_to_cartesian=True, wrap_around_amount=1):
#         from measurements.constants import EARTH_RADIUS, MAX_SEA_DEPTH
#         
#         self.EARTH_RADIUS = EARTH_RADIUS
#         self.convert_spherical_to_cartesian = convert_spherical_to_cartesian
#         
#         #TODO  move in seperate "EARTH" interpolator
#         
#         ## estimate deviation
# #         if separation_values == SEPARATION_VALUES and minimum_measurements == MIN_MEASUREMENTS and measurements_file == MEASUREMENTS_DICT_FILE:
# #             deviation = estimation.load_deviations_from_measurements()
# #         else:
# #             deviation = estimation.deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file, t_range=t_range, x_range=x_range)
# #         deviation = estimation.deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file, t_range=t_range, x_range=x_range)
#         deviation = estimation.deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file)
#         
#         logging.debug('Got standard deviation estimation at %d points.' % deviation.shape[0])
#         
#         assert np.min(deviation[0,:]) >= t_range[0] and np.max(deviation[0,:]) <= t_range[1]
#         assert np.min(deviation[1,:]) >= x_range[0] and np.max(deviation[1,:]) <= x_range[1]
#         
#         
#         ## append values for lower and upper bound of depth
#         lower_depth_bound = np.min(deviation[3,:])
#         upper_depth_bound = np.max(deviation[3,:])
#         
#         assert lower_depth_bound >= 0 and upper_depth_bound <= MAX_SEA_DEPTH
#         
#         if lower_depth_bound > 0:
#             lower_depth_bound_values = deviation[deviation[3,:] == lower_depth_bound]
#             lower_depth_bound_values[3,:] = 0
#         if upper_depth_bound < MAX_SEA_DEPTH:
#             upper_depth_bound_values = deviation[deviation[3,:] == upper_depth_bound]
#             upper_depth_bound_values[3,:] = MAX_SEA_DEPTH
#         
#         deviation = np.concatenate((lower_depth_bound_values, deviation, upper_depth_bound_values), axis=0)
#         
#         
#         
#         ## wrap around
# #         t_len = t_range[1] - t_range[0]
# #         x_len = x_range[1] - x_range[0]
#         
#         deviation = util.interpolate.wrap_around(deviation, 0, t_len, amount=wrap_around_amount)
#         if not convert_spherical_to_cartesian:
#             deviation = util.interpolate.wrap_around(deviation, 1, x_len, amount=wrap_around_amount)
#         
#         
#         ## calculate t scaling factor
#         if convert_spherical_to_cartesian:
#             t_scaling_factor = 4 * EARTH_RADIUS / t_len
#         else:
#             t_scaling_factor = x_len / t_len
#         self.t_scaling_factor = t_scaling_factor
#         
#         
# #         ## sort deviation by t
# #         deviation = deviation[np.argsort(deviation[:,0])]
#         
#         
#         ## split deviation in points and values
#         deviation_points = deviation[:, :-1]
#         deviation_values = deviation[:, -1]
#         
#         self.deviation_points = self.convert_points(deviation_points)
#         self.deviation_values = deviation_values
#     
#     
#     def convert_points(self, points):
# #         t_scaling_factor = self.t_scaling_factor
# #         
# #         logging.debug('Scaling t by factor {}.'.format(t_scaling_factor))
# #         points[:,0] *= t_scaling_factor
#         
#         if self.convert_spherical_to_cartesian:
#             points[:,1:] = util.spherical.to_cartesian(points[:,1:], surface_radius=self.EARTH_RADIUS)
#         
#         return points
#     
#     
#     def deviation(self, interpolation_points, number_of_linear_interpolators=1, total_overlapping_linear_interpolators=0):
#         logging.debug('Interpolating standard deviation for {} points.'.format(interpolation_points.shape[0]))
#         
#         ## prepare interpolation points
#         # copy points
#         interpolation_points = np.array(interpolation_points, copy=True)
#         
#         # make 2 dim
#         if interpolation_points.ndim == 1:
#             interpolation_points = interpolation_points[None, :]
#         
#         # discard year
#         interpolation_points[:, 0] = interpolation_points[:, 0] % 1
#         
#         # convert to cartesian
#         interpolation_points = self.convert_points(interpolation_points)
#         
#         ## interpolating
#         deviation = util.interpolate.interpolate(self.deviation_points, self.deviation_values, interpolation_points, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, scaling_values=[self.t_scaling_factor, 1, 1, 1])
#         
#         return deviation
#     
#     