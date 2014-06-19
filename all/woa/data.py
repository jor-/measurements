import numpy as np

import logging
logger = logging.getLogger(__name__)

import measurements.dop.woa.io
import measurements.po4.woa.data.io
import ndop.model.data
import util.io



def npy_or_save_dop_and_po4(npy_file, dop_function, po4_function):
    try:
        logger.debug('Loading data from {}.'.format(npy_file))
        
        data = np.load(npy_file)
        
    except (OSError, IOError):
        logger.debug('File {} does not exists. Calculating data.'.format(npy_file))
        
        dop = dop_function()
        po4 = po4_function()
        dop = dop.reshape((1,) + dop.shape)
        po4 = po4.reshape((1,) + po4.shape)
        
        data = np.append(dop, po4, axis=0)
        util.io.save_npy(data, npy_file, make_read_only=True, create_path_if_not_exists=True)
    
    return data


def nobs():
    from .constants import NOBS_FILE
    data = npy_or_save_dop_and_po4(NOBS_FILE, measurements.dop.woa.io.nobs, measurements.po4.woa.data.io.nobs)
    
    assert np.all(data[np.logical_not(np.isnan(data))] >= 0)
    
    return data


def means():
    from .constants import MEANS_FILE
    data = npy_or_save_dop_and_po4(MEANS_FILE, measurements.dop.woa.io.means, measurements.po4.woa.data.io.means)
    
    assert np.all(data[np.logical_not(np.isnan(data))] >= 0)
    
    return data


def varis():
    from .constants import VARIS_FILE
    data = npy_or_save_dop_and_po4(VARIS_FILE, measurements.dop.woa.io.varis, measurements.po4.woa.data.io.varis)
    
    assert np.all(data[np.logical_not(np.isnan(data))] > 0)
    
    return data
