import numpy as np
import logging

import util.petsc.universal


def _check_land_sea_mask(land_sea_mask):
    ## check input
    if land_sea_mask.ndim != 2:
        raise ValueError('The land sea mask must have 2 dimensions, but its shape is {}.'.format(land_sea_mask.shape))


def resolution_128x64x15():
    from .constants import LSM_128x64x15_PETSC_FILE, LSM_128x64x15_NPY_FILE
    
    try:
        land_sea_mask = np.load(LSM_128x64x15_NPY_FILE)
        
        logging.debug('Returning land-sea-mask loaded from {} file.'.format(LSM_128x64x15_NPY_FILE))
        
    except (OSError, IOError):
        land_sea_mask = util.petsc.universal.load_petsc_mat_to_array(LSM_128x64x15_PETSC_FILE, dtype=int)
        land_sea_mask = land_sea_mask.transpose() # metos3d: x and y are changed
        
        logging.debug('Saving land-sea-mask to {} file.'.format(LSM_128x64x15_NPY_FILE))
        
        np.save(LSM_128x64x15_NPY_FILE, land_sea_mask)
        
        logging.debug('Returning land-sea-mask loaded from petsc file.')
    
    _check_land_sea_mask(land_sea_mask)
    
    return land_sea_mask