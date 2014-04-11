import numpy as np

import logging
logger = logging.getLogger(__name__)

import ndop.model.data
import measurements.dop.woa.regrid


def load_npy_or_save(npy_file):
    try:
        logger.debug('Loading data from {}.'.format(npy_file))
        data = np.load(npy_file)
    except (OSError, IOError):
        logger.debug('File {} does not exists. Calculating DOP data.'.format(npy_file))
        land_sea_mask = ndop.model.data.load_land_sea_mask()
        measurements.dop.woa.regrid.save(land_sea_mask, t_dim=12)
        data = np.load(npy_file)
    
    return data



def nobs():
    from measurements.dop.woa.constants import NOBS_FILE
    data = load_npy_or_save(NOBS_FILE)
    return data


def varis():
    from measurements.dop.woa.constants import VARIS_FILE
    data = load_npy_or_save(VARIS_FILE)
    return data


def means():
    from measurements.dop.woa.constants import MEANS_FILE
    data = load_npy_or_save(MEANS_FILE)
    return data