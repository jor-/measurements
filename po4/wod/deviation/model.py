import numpy as np
import logging

import measurements.util.interpolate
from . import estimation



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
