import os.path

from measurements.all.constants import BASE_DIR

WOA_BASE_DIR = os.path.join(BASE_DIR, 'woa')

NOBS_FILE =    os.path.join(WOA_BASE_DIR, 'dop_po4_2.8x2.8_monthly_nobs.npy')
VARIS_FILE =   os.path.join(WOA_BASE_DIR, 'dop_po4_2.8x2.8_monthly_vari.npy')
MEANS_FILE =   os.path.join(WOA_BASE_DIR, 'dop_po4_2.8x2.8_monthly_mean.npy')