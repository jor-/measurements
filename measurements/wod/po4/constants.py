import measurements.land_sea_mask.lsm


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

SAMPLE_T_DIM = 12
SAMPLE_LSM = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R(t_dim=SAMPLE_T_DIM)

STANDARD_DEVIATION_MIN_VALUE = 0.1
CORRELATION_MIN_MEASUREMENTS = 40

FILL_STRATEGY = 'interpolate'
