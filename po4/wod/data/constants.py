import datetime
import os.path

from ..constants import WOD_DIR, ANALYSIS_DIR

DATA_DIR = os.path.join(WOD_DIR, 'data', 'cruises')
CRUISES_FILE = os.path.join(ANALYSIS_DIR, 'cruises_list.ppy')
MEASUREMENTS_DICT_FILE = os.path.join(ANALYSIS_DIR, 'measurement_dict.ppy')
MEASUREMENTS_DICT_UNSORTED_FILE = os.path.join(ANALYSIS_DIR, 'measurement_dict_unsorted.ppy')
MEASUREMENTS_POINTS_FILE = os.path.join(ANALYSIS_DIR, 'measurement_points.npy')
MEASUREMENTS_RESULTS_FILE = os.path.join(ANALYSIS_DIR, 'measurement_results.npy')

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