import bisect
import logging
import numpy as np

import measurements.po4.woa.data13.load
import measurements.util.regrid

import util.math.interpolate



def measurements_to_land_sea_mask(land_sea_mask, z_values):
    from .constants import T_RANGE, X_RANGE, Y_RANGE, ANNUAL_THRESHOLD
    from .constants import VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR, VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR, VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND
    from measurements.po4.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE

    ## load measurement data and regrid them
    measurement_data = measurements.po4.woa.data13.load.measurement_data()
    (means, nobs, variances) = measurements.util.regrid.measurements_to_land_sea_mask(measurement_data, land_sea_mask, t_range=T_RANGE, x_range=X_RANGE, y_range=Y_RANGE)
    assert means.ndim == 4 and nobs.ndim == 4 and variances.ndim == 4
    assert means.shape == nobs.shape == variances.ndim

    ## remove variances with to few measurements
    ANNUAL_THRESHOLD_INDEX = bisect.bisect_right(z_values, ANNUAL_THRESHOLD)
    variances[:,:,:,:ANNUAL_THRESHOLD_INDEX][nobs[:,:,:,:ANNUAL_THRESHOLD_INDEX] < DEVIATION_MIN_MEASUREMENTS] = np.inf
    variances[:,:,:,ANNUAL_THRESHOLD_INDEX:][nobs[:,:,:,ANNUAL_THRESHOLD_INDEX:] < DEVIATION_MIN_MEASUREMENTS / land_sea_mask.t_dim] = np.inf

    ## set variances below lower bound to lower bound
    variances[variances < DEVIATION_MIN_VALUE**2] = DEVIATION_MIN_VALUE**2


    ## interpolate missing variances
    data_index_t, data_index_x, data_index_y, data_index_z = np.where(nobs >= DEVIATION_MIN_MEASUREMENTS)
    data_points_indices = (data_index_t, data_index_x, data_index_y, data_index_z)
    data_points = np.array(data_points_indices).swapaxes(0,1)
    data_values = variances[data_points_indices]

    interpolation_index_t, interpolation_index_x, interpolation_index_y, interpolation_index_z = np.where(nobs < DEVIATION_MIN_MEASUREMENTS)
    interpolation_points_indices = (interpolation_index_t, interpolation_index_x, interpolation_index_y, interpolation_index_z)
    interpolation_points = np.array(interpolation_points_indices).swapaxes(0,1)

    logging.debug('Interpolating variance for {} points.'.format(len(interpolation_points)))

    interpolator = util.math.interpolate.Periodic_Interpolator(data_points, data_values, point_range_size=variances.shape, wrap_around_amount=VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR, single_overlapping_amount_linear_interpolators=VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR/VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR)

    variances[interpolation_points_indices] = interpolator.interpolate(interpolation_points)

    assert np.all(np.isfinite(np.logical_not(np.isnan(variances))))


    ## return regrided values
    return (means, nobs, variances)

