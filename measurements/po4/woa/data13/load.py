import numpy as np

import util.io.netcdf
import measurements.po4.woa.data13.save

import util.logging
logger = util.logging.logger


## (raw) data from woa

def woa_data(data_name, divide_annual_values):
    from .constants import ANNUAL_FILE, MONTHLY_FILES

    def data_from_file(file, data_name):
        return util.io.netcdf.load(file, data_name).swapaxes(1,3)
        assert data.ndim == 4
        assert data.shape[:3] == (1, 360, 180)

    ## load annual data
    t_dim = len(MONTHLY_FILES)
    data = data_from_file(ANNUAL_FILE, data_name)
    # data = np.tile(data.T, t_dim).T
    data = np.repeat(data, t_dim, axis=0)
    if divide_annual_values:
        data = data / t_dim

    ## load monthly data
    for t in range(t_dim):
        data_monthly = data_from_file(MONTHLY_FILES[t], data_name)
        z_dim = data_monthly.shape[-1]
        data[t, :, :, :z_dim] = data_monthly

    ## return
    return data


def depth_values():
    from .constants import ANNUAL_FILE, DEPTH_DATANAME
    return util.io.netcdf.load(ANNUAL_FILE, DEPTH_DATANAME)


def measurement_data():
    from .constants import MEAN_DATANAME, NOBS_DATANAME, STANDARD_DEVIATION_DATANAME

    ## load data
    means = woa_data(MEAN_DATANAME, False).astype(np.float64)
    nobs = woa_data(NOBS_DATANAME, True)
    deviations = woa_data(STANDARD_DEVIATION_DATANAME, False).astype(np.float64)

    ## calculate sum of values and squares
    sum_of_values = means * nobs
    sum_of_squares = deviations**2 * (nobs - 1) + means**2

    mask = nobs == 1 & ~ means.mask
    sum_of_squares[mask] = means[mask]**2

    ## get measurement points
    mask = np.where(nobs > 0)
    points = np.array([mask_i.data for mask_i in mask]).swapaxes(0, 1)

    ## center months
    points[:, 0] += 0.5

    ## get corresponding depths
    depth = depth_values()
    points[:, 3] = depth[points[:, 3]]

    ### concatenate data
    measurement_data = np.concatenate([points, sum_of_values[mask].data[:,np.newaxis], nobs[mask].data[:,np.newaxis], sum_of_squares[mask].data[:,np.newaxis]], axis=1)

    return measurement_data



## regrided data

def npy_or_save(npy_file):
    try:
        logger.debug('Loading data from {}.'.format(npy_file))
        data = np.load(npy_file)
    except (OSError, IOError):
        logger.debug('File {} does not exists. Calculating PO4 woa data.'.format(npy_file))
        measurements.po4.woa.data13.save.measurements_to_metos_boxes()
        data = np.load(npy_file)

    return data


def nobs():
    from .constants import NOBS_FILE
    data = npy_or_save(NOBS_FILE)
    return data


def variances():
    from .constants import VARIANCES_FILE
    data = npy_or_save(VARIANCES_FILE)
    return data


def means():
    from .constants import MEANS_FILE
    data = npy_or_save(MEANS_FILE)
    return data
