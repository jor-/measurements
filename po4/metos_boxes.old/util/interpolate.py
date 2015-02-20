import numpy as np

import measurements.util.map
# import measurements.util.interpolate

from ndop.model.constants import METOS_SPACE_DIM, LSM

import util.math.interpolate

import util.logging
logger = util.logging.get_logger()



def metos_boxes(data_points, data_values, t_dim, interpolator_setup, parallel=False):
    ## unpack input
    assert len(interpolator_setup) == 3
    amount_of_wrap_around, number_of_linear_interpolators, total_overlapping_linear_interpolators = interpolator_setup
    
    ## prepare amount_of_wrap_around
    try:
        amount_of_wrap_around = tuple(amount_of_wrap_around)
    except TypeError:
        amount_of_wrap_around = (amount_of_wrap_around,)
    if len(amount_of_wrap_around) == 1:
        ## use same wrap around for t and x
        amount_of_wrap_around = amount_of_wrap_around * 2
    if len(amount_of_wrap_around) == 2:
        ## append wrap around for y and z if missing
        amount_of_wrap_around = amount_of_wrap_around + (0,0)
    
    ## prepare interpolation points
    masked_map = measurements.util.map.init_masked_map(LSM)
    sea_indices = np.array(np.where(np.logical_not(np.isnan(masked_map)))).transpose()
    sea_indices_len = sea_indices.shape[0]
    logger.debug('Found {} sea points.'.format(sea_indices_len))
    logger.debug('Preparing {} interpolation points for t_dim {}.'.format(t_dim * sea_indices_len, t_dim))
    interpolation_points = np.empty((t_dim * sea_indices_len, sea_indices.shape[1] + 1))
    for t in range(t_dim):
        interpolation_points[t*sea_indices_len : (t+1)*sea_indices_len, 0] = float(t) / t_dim
        interpolation_points[t*sea_indices_len : (t+1)*sea_indices_len, 1:] = sea_indices
    
    
    ## interpolate
    T_LEN = 1
#     interpolator = measurements.util.interpolate.Time_Periodic_Non_Cartesian_Interpolator(data_points, data_values, t_len=t_dim, x_len=METOS_X_DIM, wrap_around_amount=amount_of_wrap_around, number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, parallel=parallel)
    interpolator = util.math.interpolate.Periodic_Interpolater(data_points, data_values, point_range_size=(T_LEN,)+METOS_SPACE_DIM, wrap_around_amount=amount_of_wrap_around, scaling_values=(METOS_SPACE_DIM[0]/T_LEN, None, None, None), number_of_linear_interpolators=number_of_linear_interpolators, total_overlapping_linear_interpolators=total_overlapping_linear_interpolators, parallel=parallel)
    
    interpolated_values = interpolator.interpolate(interpolation_points)
    
    return (interpolation_points, interpolated_values)

