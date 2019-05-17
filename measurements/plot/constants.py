import os.path

from measurements.constants import BASE_DIR

# base dir

BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'plots')

MEASUREMENT_DIR = os.path.join(BASE_DIR, 'measurements', '{tracer}', '{data_set}')

KIND_DIR = os.path.join(MEASUREMENT_DIR, '{kind}', '{kind_id}')

PLOT_DEFAULT_FILE_EXTENSION = 'svg'
PLOT_FILE_WITHOUT_FILE_EXTENSION = os.path.join(KIND_DIR, '{plot_name}')
PLOT_FILE = PLOT_FILE_WITHOUT_FILE_EXTENSION + '.{file_extension}'
