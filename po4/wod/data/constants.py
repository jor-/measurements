import datetime

BASE_DATE = datetime.datetime(1770, 1, 1)

## netcdf datanames
DAY_OFFSET = 'time' # number of days since 01.01.1770 (float)
LAT = 'lat'
LON = 'lon'
DEPTH = 'z'
DEPTH_FLAG = 'z_WODflag'
PO4 = 'Phosphate'
PO4_FLAG = 'Phosphate_WODflag'
PO4_PROFILE_FLAG = 'Phosphate_WODprofileflag'
MISSING_VALUE = - 10**10