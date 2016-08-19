import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..data.results import Measurements

from ..data.constants import MEASUREMENTS_DICT_SORTED_FILE
from .constants import SEPARATION_VALUES_NORMALIZED, MIN_MEASUREMENTS_NORMALIZED, MEASUREMENTS_NORMALIZED_DICT_FILE


def save_normalized_measurement_dict(dict_file=MEASUREMENTS_DICT_SORTED_FILE, normalized_dict_file=MEASUREMENTS_NORMALIZED_DICT_FILE):
    m = Measurements()
    m.load(dict_file)
    m.normalize(SEPARATION_VALUES_NORMALIZED, minimum_measurements=MIN_MEASUREMENTS_NORMALIZED)
    m.save(normalized_dict_file)