from measurements.universal.constants import SAMPLE_LSM

# informations for the wod cruises

DATA_VARIABLE = {
    'name_prefix': 'Taxon_count_',
    'unit': b'#/m3',
    'unit_name': 'CBV_units',
    'flag_name': 'CBV_flag',
    'flag_valid_value': 0,
    'comment': b'contains two values: first value is value as originally measured or observed.  Second values is the common biological value as converted to common units for comparison.',
    'comment_name': 'comment',
    'plankton_group_code_name': 'PGC_group_code',
    'upper_depth_name': 'upper_depth',
    'lower_depth_name': 'lower_depth'}

PHYTOPLANKTON_GROUP_CODE = 2000000
ZOOPLANKTON_GROUP_CODE = 4000000


# sample informations

TRACERS = ('zooplankton', 'phytoplankton')

STANDARD_DEVIATION_MIN_VALUE = 0.5
CORRELATION_MIN_MEASUREMENTS = 40

FILL_STRATEGY = 'point_average'
