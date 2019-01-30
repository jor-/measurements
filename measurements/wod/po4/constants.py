from measurements.universal.constants import SAMPLE_LSM

# informations for the wod cruises

DATA_VARIABLE = {
    'name': 'Phosphate',
    'unit': b'umol/l',
    'unit_name': 'units',
    'missing_data_value': - 10**10,
    'flag_name': 'Phosphate_WODflag',
    'flag_valid_value': 0,
    'profile_flag_name': 'Phosphate_WODprofileflag',
    'profile_flag_valid_value': 0}

DEPTH_VARIABLE = {
    'name': 'z',
    'unit': b'm',
    'unit_name': 'units',
    'flag_name': 'z_WODflag',
    'flag_valid_value': 0}


# sample informations

TRACER = 'po4'

STANDARD_DEVIATION_MIN_VALUE = 0.1
CORRELATION_MIN_MEASUREMENTS = 40

FILL_STRATEGY = 'interpolate'
