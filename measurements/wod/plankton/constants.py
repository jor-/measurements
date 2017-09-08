import measurements.land_sea_mask.lsm


# informations for the wod cruises

DATA_NAME_PREFIX = 'Taxon_count_'

DATA_UNIT = b'#/m3'
DATA_UNIT_NAME = 'CBV_units'
DATA_COMMENT = b'contains two values: first value is value as originally measured or observed.  Second values is the common biological value as converted to common units for comparison.'
DATA_COMMENT_NAME = 'comment'

DATA_FLAG_NAME = 'CBV_flag'
DATA_VALID_FLAG = 0

DATA_PLANKTON_GROUPE_CODE_NAME = 'PGC_group_code'

DATA_UPPER_DEPTH_NAME = 'upper_depth'
DATA_LOWER_DEPTH_NAME = 'lower_depth'

PHYTOPLANKTON_GROUP_CODE = 2000000
ZOOPLANKTON_GROUP_CODE = 4000000


# sample informations

TRACERS = ('zooplankton', 'phytoplankton')

SAMPLE_T_DIM = 12
SAMPLE_LSM = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R(t_dim=SAMPLE_T_DIM)

STANDARD_DEVIATION_MIN_VALUE = 0.5
CORRELATION_MIN_MEASUREMENTS = 40

FILL_STRATEGY = 'point_average'
