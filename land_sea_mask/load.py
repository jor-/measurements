import numpy as np

import measurements.land_sea_mask.constants
import util.petsc.universal

import util.logging
logger = util.logging.logger


def _check_land_sea_mask(land_sea_mask):
    ## check input
    if land_sea_mask.ndim != 2:
        raise ValueError('The land sea mask must have 2 dimensions, but its shape is {}.'.format(land_sea_mask.shape))


def resolution_128x64x15():
    from .constants import LSM_128x64x15_PETSC_FILE, LSM_128x64x15_NPY_FILE

    try:
        land_sea_mask = np.load(LSM_128x64x15_NPY_FILE)

        logger.debug('Returning land-sea-mask loaded from {} file.'.format(LSM_128x64x15_NPY_FILE))

    except (OSError, IOError):
        land_sea_mask = util.petsc.universal.load_petsc_mat_to_array(LSM_128x64x15_PETSC_FILE, dtype=int)
        land_sea_mask = land_sea_mask.transpose() # metos3d: x and y are changed

        _check_land_sea_mask(land_sea_mask)
        logger.debug('Saving land-sea-mask to {} file.'.format(LSM_128x64x15_NPY_FILE))
        np.save(LSM_128x64x15_NPY_FILE, land_sea_mask)
        logger.debug('Returning land-sea-mask loaded from petsc file.')


    return land_sea_mask


def resolution_128x64x15_normalized_volumes():
    from .constants import LSM_128x64x15_NORMALIZED_VLUMES_PETSC_FILE
    logger.debug('Loading normalized valumes from petsc file {}.'.format(LSM_128x64x15_NORMALIZED_VLUMES_PETSC_FILE))
    return util.petsc.universal.load_petsc_vec_to_numpy_array(LSM_128x64x15_NORMALIZED_VLUMES_PETSC_FILE)



## 60x180x138

def resolution_360x180x138():
    from .constants import LSM_360x180x138_TXT_FILE, LSM_360x180x138_NPY_FILE

    try:
        lsm = np.load(LSM_360x180x138_NPY_FILE)
        logger.debug('Returning land-sea-mask loaded from {} file.'.format(LSM_360x180x138_NPY_FILE))

    except (OSError, IOError):
        ## read values from txt with axis order: x y z
        lsm = np.genfromtxt(LSM_360x180x138_TXT_FILE, dtype=float, delimiter=',', comments='#', usecols=(1, 0, 2))

        ## normalize values
        min_values = lsm.min(axis=0)
        lsm[:,0] = lsm[:,0] - min_values[0]
        lsm[:,1] = lsm[:,1] - min_values[1]
        lsm[:,2] = lsm[:,2] - 1

        ## convert to int
        lsm_int = lsm.astype(np.int16)

        assert np.all(lsm_int == lsm)
        assert lsm_int[:,0].min() == 0 and lsm_int[:,0].max() == 359 and lsm_int[:,1].min() == 0 and lsm_int[:,1].max() == 179 and lsm_int[:,2].min() == 0 and lsm_int[:,2].max() == 137

        ## convert in 2 dim
        lsm = np.empty((360, 180), dtype=np.int16)
        for x, y, z in lsm_int:
            lsm[x, y] = z
        assert lsm.min() == 0 and lsm.max() == 137

        ## save lsm as npy
        _check_land_sea_mask(lsm)
        logger.debug('Saving land-sea-mask to {} file.'.format(LSM_360x180x138_NPY_FILE))
        np.save(LSM_360x180x138_NPY_FILE, lsm)
        logger.debug('Returning land-sea-mask loaded from txt file.')

    return lsm


def depth_360x180x138():
    from .constants import DEPTH_360x180x138_TXT_FILE, DEPTH_360x180x138_NPY_FILE

    ## read from cache
    try:
        depth = measurements.land_sea_mask.constants.DEPTH_360x180x138
        logger.debug('Returning depth levels from cache.')

    ## read from npy file
    except AttributeError:
        try:
            depth = np.load(DEPTH_360x180x138_NPY_FILE)
            logger.debug('Returning depth levels loaded from {} file.'.format(DEPTH_360x180x138_NPY_FILE))
            measurements.land_sea_mask.constants.DEPTH_360x180x138 = depth

    ## read from txt file
        except (OSError, IOError):
            ## read values from txt with axis order: x y z
            depth = np.genfromtxt(DEPTH_360x180x138_TXT_FILE, dtype=np.int16, comments='#', usecols=(0,))
            assert depth.ndim == 1 and depth.shape[0] == 138

            ## save depth as npy
            logger.debug('Saving depth to {} file.'.format(DEPTH_360x180x138_NPY_FILE))
            np.save(DEPTH_360x180x138_NPY_FILE, depth)
            logger.debug('Returning land-sea-mask loaded from txt file.')
            measurements.land_sea_mask.constants.DEPTH_360x180x138 = depth

    return depth
