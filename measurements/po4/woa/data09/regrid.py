import numpy as np
import bisect
import logging

import measurements.land_sea_mask.lsm

import util.io.np
import util.io.netcdf
import util.io.fs
import util.math.interpolate


def load_from_netcdf(netcdf_file, netcdf_dataname):
    logging.debug('Loading data from {}.'.format(netcdf_file))

    data = util.io.netcdf.load(netcdf_file, netcdf_dataname)
    data = np.swapaxes(data, 1, 3)  # netcdf shape: (12, 15, 64, 128)

    return data


def save():
    from .constants import NOBS_NETCDF_ANNUAL_FILE, NOBS_NETCDF_MONTHLY_FILE, NOBS_NETCDF_DATANAME, NOBS_FILE
    from .constants import VARIS_NETCDF_ANNUAL_FILE, VARIS_NETCDF_MONTHLY_FILE, VARIS_NETCDF_DATANAME, VARIS_FILE
    from .constants import MEANS_NETCDF_ANNUAL_FILE, MEANS_NETCDF_MONTHLY_FILE, MEANS_NETCDF_DATANAME, MEANS_FILE
    from .constants import ANNUAL_THRESHOLD, VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR, VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR, VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND
    from measurements.po4.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE


    ## concatenate annual and montly WOA data
    METOS_Z_LEFT = measurements.land_sea_mask.lsm.LandSeaMaskTMM().z_left
    
    z_index_annual_threshold = bisect.bisect_right(METOS_Z_LEFT, ANNUAL_THRESHOLD)
    logging.debug('Taking annual data from z index {}.'.format(z_index_annual_threshold))

    for (netcdf_annual_file, netcdf_monthly_file, netcdf_dataname, npy_file, divide) in ((NOBS_NETCDF_ANNUAL_FILE, NOBS_NETCDF_MONTHLY_FILE, NOBS_NETCDF_DATANAME, NOBS_FILE, True), (VARIS_NETCDF_ANNUAL_FILE, VARIS_NETCDF_MONTHLY_FILE, VARIS_NETCDF_DATANAME, VARIS_FILE, False), (MEANS_NETCDF_ANNUAL_FILE, MEANS_NETCDF_MONTHLY_FILE, MEANS_NETCDF_DATANAME, MEANS_FILE, False)):

        logging.debug('Preparing {}.'.format(npy_file))

        data_monthly = load_from_netcdf(netcdf_monthly_file, netcdf_dataname)
        data_annual = load_from_netcdf(netcdf_annual_file, netcdf_dataname)

        t_dim = data_monthly.shape[0]
        if divide:
            factor = 1 / t_dim
        else:
            factor = 1

        for t in range(t_dim):
            data_monthly[t,:,:,z_index_annual_threshold:] = data_annual[0,:,:,z_index_annual_threshold:] * factor

        util.io.np.save(npy_file, data_monthly, make_read_only=True, create_path_if_not_exists=True)


    ## revise variance
    nobs = np.load(NOBS_FILE)
    vari = np.load(VARIS_FILE)
    variance_min_value = DEVIATION_MIN_VALUE**2
    vari[vari < variance_min_value] = variance_min_value

    data_index_t, data_index_x, data_index_y, data_index_z = np.where(nobs >= DEVIATION_MIN_MEASUREMENTS)
    data_points_indices = (data_index_t, data_index_x, data_index_y, data_index_z)
    data_points = np.array(data_points_indices).swapaxes(0,1)
    data_values = vari[data_points_indices]

    assert data_points.ndim == 2
    assert data_values.ndim == 1
    assert len(data_points) == len(data_values)


    ## interpolate variance
    interpolation_index_t, interpolation_index_x, interpolation_index_y, interpolation_index_z = np.where(nobs < DEVIATION_MIN_MEASUREMENTS)
    interpolation_points_indices = (interpolation_index_t, interpolation_index_x, interpolation_index_y, interpolation_index_z)
    interpolation_points = np.array(interpolation_points_indices).swapaxes(0,1)

    logging.debug('Interpolating variance for {} points.'.format(interpolation_points.shape[0]))

    interpolator = util.math.interpolate.Periodic_Interpolator(data_points, data_values, point_range_size=vari.shape, wrap_around_amount=VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR, single_overlapping_amount_linear_interpolators=VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR/VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR)

    vari[interpolation_points_indices] = interpolator.interpolate(interpolation_points)

    assert np.all(np.isfinite(np.logical_not(np.isnan(vari))))


    ## saving interpolated variance
    logging.debug('Saving interpolated variance.')
    util.io.fs.make_writable(VARIS_FILE)
    util.io.np.save(VARIS_FILE, vari, make_read_only=True, create_path_if_not_exists=True)

