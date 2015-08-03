import os.path

from measurements.constants import BASE_DIR
import measurements.land_sea_mask.data

WOD_DIR = os.path.join(BASE_DIR, 'po4/wod13')
ANALYSIS_DIR = os.path.join(WOD_DIR, 'analysis')

SAMPLE_LSM = measurements.land_sea_mask.data.LandSeaMaskWOA13R(t_dim=48)
