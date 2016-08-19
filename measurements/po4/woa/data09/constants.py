import os.path

from measurements.constants import BASE_DIR


## PO4
WOA_BASE_DIR = os.path.join(BASE_DIR, 'po4/woa09')
NOBS_NETCDF_MONTHLY_FILE = os.path.join(WOA_BASE_DIR, 'woa09_po4_2.8x2.8_monthly_nobs.cdf')
NOBS_NETCDF_ANNUAL_FILE = os.path.join(WOA_BASE_DIR, 'woa09_po4_2.8x2.8_annual_nobs.cdf')
NOBS_NETCDF_DATANAME = 'PO4NOBS'

VARIS_NETCDF_MONTHLY_FILE = os.path.join(WOA_BASE_DIR, 'woa09_po4_2.8x2.8_monthly_vari.cdf')
VARIS_NETCDF_ANNUAL_FILE = os.path.join(WOA_BASE_DIR, 'woa09_po4_2.8x2.8_annual_vari.cdf')
VARIS_NETCDF_DATANAME = 'PO4VARI'

MEANS_NETCDF_MONTHLY_FILE = os.path.join(WOA_BASE_DIR, 'woa09_po4_2.8x2.8_monthly_mean.cdf')
MEANS_NETCDF_ANNUAL_FILE = os.path.join(WOA_BASE_DIR, 'woa09_po4_2.8x2.8_annual_mean.cdf')
MEANS_NETCDF_DATANAME = 'PO4MEAN'

NOBS_FILE = os.path.join(WOA_BASE_DIR, 'po4_2.8x2.8_monthly_nobs.npy')
VARIS_FILE = os.path.join(WOA_BASE_DIR, 'po4_2.8x2.8_monthly_vari.npy')
MEANS_FILE = os.path.join(WOA_BASE_DIR, 'po4_2.8x2.8_monthly_mean.npy')

ANNUAL_THRESHOLD = 500

VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR = 1
VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0
VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND = (0.1, 0.1, 0, 0)


#
# ## DOP
# YOSHIMURA_DOP_MEASUREMENT_FILE = os.path.join(BASE_DIR, 'dop/Yoshimura2007/Yoshimura2007_prepared.txt')
# LADOLFI_2002_DOP_MEASUREMENT_FILE = os.path.join(BASE_DIR, 'dop/Ladolfi2002/CD139_DOP_prepared.txt')
# LADOLFI_2004_DOP_MEASUREMENT_FILE = os.path.join(BASE_DIR, 'dop/Ladolfi2004/D279_DOP_prepared.txt')
#
# DOP_NOBS = os.path.join(BASE_DIR, 'dop/dop_2.8x2.8_monthly_nobs.npy')
# DOP_VARIS = os.path.join(BASE_DIR, 'dop/dop_2.8x2.8_monthly_vari.npy')
# DOP_MEANS = os.path.join(BASE_DIR, 'dop/dop_2.8x2.8_monthly_mean.npy')
#
#
# ## BOTH
# NOBS = os.path.join(BASE_DIR, 'dop_po4_2.8x2.8_monthly_nobs.npy')
# VARIS = os.path.join(BASE_DIR, 'dop_po4_2.8x2.8_monthly_vari.npy')
# MEANS = os.path.join(BASE_DIR, 'dop_po4_2.8x2.8_monthly_mean.npy')

