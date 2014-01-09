import numpy as np
import scipy.interpolate
import logging

from . import estimation
from .constants import MIN_MEASUREMENTS, SEPARATION_VALUES, T_RANGE, X_RANGE
from ..data.constants import MEASUREMENTS_DICT_FILE

import util.spherical

logger = logging.getLogger(__name__)


class Deviation_Model():
    
    def __init__(self, minimum_measurements=MIN_MEASUREMENTS, separation_values=SEPARATION_VALUES, t_range=T_RANGE, x_range=X_RANGE, measurements_file=MEASUREMENTS_DICT_FILE, convert_spherical_to_cartesian=True):
        
        self.convert_spherical_to_cartesian = convert_spherical_to_cartesian
        
        ## estimate deviation
#         if separation_values == SEPARATION_VALUES and minimum_measurements == MIN_MEASUREMENTS and measurements_file == MEASUREMENTS_DICT_FILE:
#             deviation = estimation.load_deviations_from_measurements()
#         else:
#             deviation = estimation.deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file, t_range=t_range, x_range=x_range)
        deviation = estimation.deviations_from_measurements(separation_values=separation_values, minimum_measurements=minimum_measurements, measurements_file=measurements_file, t_range=t_range, x_range=x_range)
        
        
        logger.debug('Got standard deviation estimation at %d points.' % deviation.shape[0])
        
        ## wrap around t
        t_len = t_range[1] - t_range[0]
        deviation_copy = deviation.copy()
        deviation_copy[:,0] += t_len
        deviation = np.append(deviation, deviation_copy, axis=0)
        deviation_copy[:,0] -= 2 * t_len
        deviation = np.append(deviation, deviation_copy, axis=0)
        
        ## wrap around x
        x_len = x_range[1] - x_range[0]
        deviation_copy = deviation.copy()
        deviation_copy[:,1] += x_len
        deviation = np.append(deviation, deviation_copy, axis=0)
        deviation_copy[:,1] -= 2 * x_len
        deviation = np.append(deviation, deviation_copy, axis=0)
        
        ## split deviation in points and values
        deviation_points = deviation[:, :-1]
        deviation_values = deviation[:, -1]
        
        if self.convert_spherical_to_cartesian:
            deviation_points = util.spherical.to_cartesian(deviation_points)
        self.deviation_points = deviation_points
        self.deviation_values = deviation_values
    
    
    
    def deviation(self, points):
        def interpolate_deviation(points, method):
            logger.debug('Interpolating standard deviation with method %s at %d points.' % (method, len(points)))
            interpolated_values = scipy.interpolate.griddata(self.deviation_points, self.deviation_values, points, method=method)
            
            return interpolated_values
        
        # check input
        if len(points.shape) == 1:
            n = points.shape[0]
            points = points.reshape(1, n)
        elif len(points.shape) >= 3:
            raise ValueError('Points have to be a vector or matrix array, but its shape is ' + str(points.shape) + '.')
        
        # copy points
        points = points.copy()
        
        # discard year
        points[:, 0] = points[:, 0] % 1
        
        # convert to cartesian
        if self.convert_spherical_to_cartesian:
            points = util.spherical.to_cartesian(points)
        
        # interpolate linear
        deviation = interpolate_deviation(points, 'linear')
        
        # interpolate nearest if necessary
        mask = np.isnan(deviation)
        if mask.sum() > 0:
            deviation[mask] = interpolate_deviation(points[mask],  'nearest')
        
        return deviation
