import os.path

from measurements.constants import BASE_DIR

# base dir

BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'plots')

MEASUREMENT_DIR = os.path.join(BASE_DIR, '{tracer}', '{data_set}')

KIND_DIR = os.path.join(MEASUREMENT_DIR, '{kind}', '{kind_id}')

PLOT_FILE = os.path.join(KIND_DIR, '{plot_name}.png')