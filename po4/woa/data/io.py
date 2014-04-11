import numpy as np

import logging
logger = logging.getLogger(__name__)

import measurements.po4.woa.data.regrid


def load_npy_or_save(npy_file):
    try:
        logger.debug('Loading data from {}.'.format(npy_file))
        data = np.load(npy_file)
    except (OSError, IOError):
        logger.debug('File {} does not exists. Calculating PO4 data.'.format(npy_file))
        measurements.po4.woa.data.regrid.save()
        data = np.load(npy_file)
    
    return data


def nobs():
    from .constants import NOBS_FILE
    data = load_npy_or_save(NOBS_FILE)
    return data



def varis():
    from .constants import VARIS_FILE
    data = load_npy_or_save(VARIS_FILE)
    return data



def means():
    from .constants import MEANS_FILE
    data = load_npy_or_save(MEANS_FILE)
    return data