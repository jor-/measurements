import numpy as np
import scipy.interpolate
import logging

from . import estimation
from .constants import MIN_MEASUREMENTS

import util.spherical

class Variance_Model():
    
    def __init__(self, minimum_measurements=MIN_MEASUREMENTS):
        from .constants import SEPARATION_VALUES#, X_RANGE
        
        self.logger = logging.getLogger(__name__)
        
        ## estimate variance
        variance = estimation.space_variance_from_measurements(minimum_measurements=minimum_measurements, separation_values=SEPARATION_VALUES)
        
#         ## wrap around x
#         x_len = X_RANGE[1] - X_RANGE[0]
#         
#         x_min = np.min(variance[:, 0])
#         mask_min = variance[:, 0] == x_min
#         wrap_min = variance[mask_min].copy()
#         wrap_min[:, 0] += x_len
#         
#         x_max = np.max(variance[:, 0])
#         mask_max = variance[:, 0] == x_max
#         wrap_max = variance[mask_max].copy()
#         wrap_max[:, 0] -= x_len
#         
#         variance = np.concatenate((variance, wrap_min, wrap_max))
        
        ## split variance in points and values
        variance_points = variance[:, 0:3]
        variance_values = variance[:, 3]
        
        self.variance_points = util.spherical.to_cartesian(variance_points)
        self.variance_values = variance_values
    
    
    
    def variance(self, points):
        def interpolate_variance(points, method):
            self.logger.debug('Interpolating variance with method %s at %d points.', method, len(points))
            interpolated_values = scipy.interpolate.griddata(self.variance_points, self.variance_values, points, method=method)
            
            return interpolated_values
        
        # check input
        if len(points.shape) == 1:
            n = points.shape[0]
            points = points.reshape(1, n)
        elif len(points.shape) >= 3:
            raise ValueError('Points have to be a vector or matrix array, but its shape is ' + str(points.shape) + '.')
        
        # discard time
        points = points[:, 1:4]
        
        # convert to cartesian
        points = util.spherical.to_cartesian(points)
        
        # interpolate linear
        variance = interpolate_variance(points, 'linear')
        
        # interpolate nearest if necessary
        mask = np.isnan(variance)
        if mask.sum() > 0:
            variance[mask] = interpolate_variance(points[mask],  'nearest')
        
        return variance