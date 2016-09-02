import datetime
import os.path

import measurements.land_sea_mask.lsm
from measurements.constants import BASE_DIR


## sample lsm

SAMPLE_T_DIM = 12
SAMPLE_LSM = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R(t_dim=SAMPLE_T_DIM)


## interpolation setups

INTERPOLATOR_SETUPS = {
'mean': {
    'concentration': {2: {'lsm_12_woa13r':(0.1,12,0.05,1), 'lsm_24_woa13r':(0.1,12,0.05,1)}, 3: {'lsm_12_woa13r':(0.1,6,0.1,1)}},
        },
'deviation': {
    # 'average_noise': {3: {'lsm_12_woa13r':(0.1,12,0.05,1), 'lsm_24_woa13r':(0.1,12,0.05,1), 'lsm_12_tmm':(0.1,6,0.1,1)}, 4: {'lsm_12_woa13r':(0.2,2,0.2,1)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}},
    # 'concentration': {3: {'lsm_12_woa13r':(0.1,12,0.1,1), 'lsm_24_woa13r':(0.1,12,0.1,1), 'lsm_12_tmm':(0.2,1,0.0,0)}, 4: {'lsm_12_woa13r':(0.2,1,0,0)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}}
    # 'average_noise': {3: {'lsm_12_woa13r':(0.1,12,0.05,1), 'lsm_24_woa13r':(0.1,12,0.1,1), 'lsm_12_tmm':(0.1,6,0.1,1)}, 4: {'lsm_12_woa13r':(0.2,2,0.2,1)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}},
    # 'concentration': {3: {'lsm_12_woa13r':(0.1,12,0.1,1), 'lsm_24_woa13r':(0.1,6,0.1,1), 'lsm_12_tmm':(0.2,1,0.0,0)}, 4: {'lsm_12_woa13r':(0.2,1,0,0)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}}
    'average_noise': {3: {'lsm_12_woa13r':(1,1,0,0), 'lsm_24_woa13r':(1,1,0,0), 'lsm_12_tmm':(0.1,6,0.1,1)}, 4: {'lsm_12_woa13r':(0.2,2,0.2,1)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}},
    'concentration': {3: {'lsm_12_woa13r':(1,1,0,0), 'lsm_24_woa13r':(1,1,0,0), 'lsm_12_tmm':(0.2,1,0.0,0)}, 4: {'lsm_12_woa13r':(0.2,1,0,0)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}}
    }
}


# MEAN_INTERPOLATOR_SETUPS = {
# 'concentration': {2: {'lsm_12_woa13r':(0.1,6,0.6,1)}, 3: {'lsm_12_woa13r':(0.2,3,0.6,1)}},
# }
# 
# DEVIATION_INTERPOLATOR_SETUPS = {
# 'average_noise': {3: {'lsm_12_woa13r':(0.1,6,0.6,1), 'lsm_12_tmm':(0.1,6,0.6,1)}, 4: {'lsm_12_woa13r':(0.2,2,0.4,1)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}},
# 'concentration': {3: {'lsm_12_woa13r':(0.2,3,0.6,1), 'lsm_12_tmm':(0.2,1,0.0,0)}, 4: {'lsm_12_woa13r':(0.2,1,0,0)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}}
# }


## data dirs and files

WOD_DIR = os.path.join(BASE_DIR, 'po4', 'wod_2013')
DATA_DIR = os.path.join(WOD_DIR, 'data')
CRUISES_DATA_DIR = os.path.join(DATA_DIR, 'cruises')
CRUISES_LIST_FILE = os.path.join(DATA_DIR, 'cruises_list.ppy')
POINTS_AND_RESULTS_FILE = os.path.join(DATA_DIR, 'points_and_results.npz')
MEASUREMENTS_DICT_SORTED_FILE = os.path.join(DATA_DIR, 'measurement_dict_sorted.ppy')
MEASUREMENTS_DICT_UNSORTED_FILE = os.path.join(DATA_DIR, 'measurement_dict_unsorted.ppy')


## informations for the wod cruises

BASE_DATE = datetime.datetime(1770, 1, 1)
DAY_OFFSET = 'time' # number of days since 01.01.1770 (float)
LAT = 'lat'
LON = 'lon'
DEPTH = 'z'
DEPTH_FLAG = 'z_WODflag'
PO4 = 'Phosphate'
PO4_FLAG = 'Phosphate_WODflag'
PO4_PROFILE_FLAG = 'Phosphate_WODprofileflag'
MISSING_VALUE = - 10**10