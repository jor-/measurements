import os.path

from measurements.constants import BASE_DIR

## base dir
WOA_BASE_DIR = os.path.join(BASE_DIR, 'po4/woa13')

## netcdf files
ANNUAL_FILE = os.path.join(WOA_BASE_DIR, 'data/annual/woa13_all_p00_01.nc')

MONTHLY_NUMBER_RANGE = range(1, 13)
MONTHLY_FILE_PATTERN = os.path.join(WOA_BASE_DIR, 'data/monthly/woa13_all_p{:0>2}_01.nc')
MONTHLY_FILES = []
for i in MONTHLY_NUMBER_RANGE:
    MONTHLY_FILES.append(MONTHLY_FILE_PATTERN.format(i))

## numpy files
MEANS_FILE = os.path.join(WOA_BASE_DIR, 'po4_2.8x2.8_monthly_means.npy')
NOBS_FILE = os.path.join(WOA_BASE_DIR, 'po4_2.8x2.8_monthly_nobs.npy')
VARIANCES_FILE = os.path.join(WOA_BASE_DIR, 'po4_2.8x2.8_monthly_variances.npy')

## data names
MEAN_DATANAME = 'p_mn'
NOBS_DATANAME = 'p_dd'
STANDARD_DEVIATION_DATANAME = 'p_sd'
DEPTH_DATANAME = 'depth'

## ranges
T_RANGE = [0, 11]
X_RANGE = [0, 360]
Y_RANGE = [0, 180]

## variance interpolator
VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR = 1
VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0
VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND = (0.1, 0.1, 0, 0)

## annual values
ANNUAL_THRESHOLD = 525