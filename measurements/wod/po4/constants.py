# informations for the wod cruises

DATA_NAME = 'Phosphate'
DATA_UNIT = b'umol/l'

DATA_FLAGS_NAME = 'Phosphate_WODflag'
DATA_VALID_FLAG = 0
DATA_PROFILE_FLAG_NAME = 'Phosphate_WODprofileflag'
DATA_PROFILE_VALID_FLAG = 0
MISSING_DATA_VALUE = - 10**10

DEPTH_NAME = 'z'
DEPTH_UNIT = b'm'

DEPTH_FLAGS_NAME = 'z_WODflag'
DEPTH_VALID_FLAG = 0


# sample informations

TRACER = 'po4'

from measurements.universal.constants import SAMPLE_LSM

STANDARD_DEVIATION_MIN_VALUE = 0.1
CORRELATION_MIN_MEASUREMENTS = 40

FILL_STRATEGY = 'interpolate'
