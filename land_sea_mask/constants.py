import os.path
import numpy as np

from measurements.constants import BASE_DIR

LSM_DIR = os.path.join(BASE_DIR, 'land_sea_masks')
LSM_128x64x15_PETSC_FILE = os.path.join(LSM_DIR, '128x64x15', 'landSeaMask.petsc')
LSM_128x64x15_NPY_FILE = os.path.join(LSM_DIR, '128x64x15', 'landSeaMask.npy')
LSM_128x64x15 = os.path.join(BASE_DIR, 'metos3d/landSeaMask.petsc')

LSM_128x64x15_DIM = (128, 64, 15)
LSM_128x64x15_Z_LEFT = np.array([0, 50, 120, 220, 360, 550, 790, 1080, 1420, 1810, 2250, 2740, 3280, 3870, 4510])
LSM_128x64x15_Z_RIGHT = np.array(LSM_128x64x15_Z_LEFT[1:].tolist() + [5200])
LSM_128x64x15_Z_CENTER = (LSM_128x64x15_Z_LEFT + LSM_128x64x15_Z_RIGHT) / 2