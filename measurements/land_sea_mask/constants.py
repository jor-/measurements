import os.path

from measurements.constants import BASE_DIR

# general
LSM_DIR = os.path.join(BASE_DIR, 'land_sea_masks')
LSM_NPY_FILENAME = 'land_sea_mask.npy'
DEPTH_NPY_FILENAME = 'depth_levels.npy'
VOLUMES_MAP_FILENAME = 'volumes_map_-_{dtype}.npy'
NORMALIZED_VOLUMES_WEIGHTS_MAP_FILENAME = 'normalized_volume_weights_map_-_{dtype}.npy'

# TMM (128x64x15)
TMM_DIR = os.path.join(LSM_DIR, 'TMM')
TMM_PETSC_FILE = os.path.join(TMM_DIR, 'land_sea_mask.petsc')
TMM_DEPTH_TXT_FILE = os.path.join(TMM_DIR, 'depth_levels.txt')
TMM_NORMALIZED_VOLUMES_PETSC_FILE = os.path.join(TMM_DIR, 'normalized_volumes.petsc')

# WOA13 (360x180x137)
WOA13_DIR = os.path.join(LSM_DIR, 'WOA13')
WOA13_LSM_TXT_FILE = os.path.join(WOA13_DIR, 'land_sea_mask.msk')
WOA13_DEPTH_TXT_FILE = os.path.join(WOA13_DIR, 'depth_levels.txt')

# WOA13R (360x180x33)
WOA13R_DIR = os.path.join(LSM_DIR, 'WOA13R')
