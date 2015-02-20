import os.path
import datetime

from measurements.constants import BASE_DIR


DATA_DIR = os.path.join(BASE_DIR, 'dop', 'data')

## data information
LADOLFI_2002_DIR = os.path.join(DATA_DIR, 'Ladolfi2002')
LADOLFI_2002_MEASUREMENT_FILE = os.path.join(LADOLFI_2002_DIR, 'CD139_DOP_prepared.txt')
# LADOLFI_2002_TIME_START = datetime.date(2002, 3, 1).timetuple().tm_yday / 365.0
# LADOLFI_2002_TIME_END = datetime.date(2002, 4, 15).timetuple().tm_yday / 365.0
LADOLFI_2002_START_DATE = datetime.date(2002, 3, 1)
LADOLFI_2002_END_DATE = datetime.date(2002, 4, 15)
LADOLFI_2002_VALID_DATA_FLAG = 1
# LADOLFI_2002_DATA_FILE = os.path.join(LADOLFI_2002_DIR, 'data.npy')

LADOLFI_2004_DIR = os.path.join(DATA_DIR, 'Ladolfi2004')
LADOLFI_2004_MEASUREMENT_FILE = os.path.join(LADOLFI_2004_DIR, 'D279_DOP_prepared.txt')
# LADOLFI_2004_TIME_START = datetime.date(2004, 4, 4).timetuple().tm_yday / 366.0
# LADOLFI_2004_TIME_END = datetime.date(2004, 5, 10).timetuple().tm_yday / 366.0
LADOLFI_2004_START_DATE = datetime.date(2004, 4, 4)
LADOLFI_2004_END_DATE = datetime.date(2004, 5, 10)
LADOLFI_2004_VALID_DATA_FLAG = 0
# LADOLFI_2004_DATA_FILE = os.path.join(LADOLFI_2004_DIR, 'data.npy')

YOSHIMURA_2007_DIR = os.path.join(DATA_DIR, 'Yoshimura2007')
YOSHIMURA_2007_MEASUREMENT_FILE = os.path.join(YOSHIMURA_2007_DIR, 'Yoshimura2007.txt')
# YOSHIMURA_2007_DATA_FILE = os.path.join(YOSHIMURA_2007_DIR, 'data.npy')

DATA_FILENAME = 'data.npy'


