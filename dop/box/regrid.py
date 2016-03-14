import numpy as np
import logging
logger = logging.getLogger(__name__)

import measurements.dop.pw.data
import measurements.util.regrid
import util.io.np



def save(land_sea_mask, z_values, t_dim=12):
    from measurements.dop.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE, T_RANGE, X_RANGE, Y_RANGE
    from .constants import NOBS_FILE, VARIS_FILE, MEANS_FILE

    logger.debug('Calculating and saving dop measurement data.')

    ## load measurement data
    measurement_data = measurements.dop.pw.data.data()


    ## regrid data
    (means, nobs, variances) = measurements.util.regrid.measurements_to_land_sea_mask(measurement_data, land_sea_mask, t_range=T_RANGE, x_range=X_RANGE, y_range=Y_RANGE)


    ## remove variances with to few measurements
    variances[nobs < DEVIATION_MIN_MEASUREMENTS] = np.inf

    ## set variances below lower bound to lower bound
    variances[variances < DEVIATION_MIN_VALUE**2] = DEVIATION_MIN_VALUE**2

    ## where no variances use averaged variance
    variances_averaged = variances[np.isfinite(variances)].mean()
    variances[variances == np.inf] = variances_averaged


    ## save values
    util.io.np.save(MEANS_FILE, means, make_read_only=True, create_path_if_not_exists=True)
    util.io.np.save(NOBS_FILE, nobs, make_read_only=True, create_path_if_not_exists=True)
    util.io.np.save(VARIS_FILE, variances, make_read_only=True, create_path_if_not_exists=True)