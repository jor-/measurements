import os.path
import numpy as np

import logging
logger = logging.getLogger(__name__)

import measurements.po4.wod.data.values
from measurements.po4.wod.data.results import Measurements_Unsorted as Measurements

from ndop.model.constants import METOS_T_DIM, METOS_X_DIM, METOS_Y_DIM, METOS_Z_LEFT
from .constants import METOS_BOXES_DICT_FILE



def save_measurement_boxes_dict(measurement_box_file=METOS_BOXES_DICT_FILE):
    m = measurements.po4.wod.data.values.measurement_dict_unsorted()
    m.categorize_indices((1./METOS_T_DIM,))
    m.transform_indices_to_boxes(METOS_X_DIM, METOS_Y_DIM, METOS_Z_LEFT)
    m.save(measurement_box_file)
    logger.debug('Measurement metos box dict saved at {}.'.format(measurement_box_file))


def load_measurement_boxes_dict(measurement_box_file=METOS_BOXES_DICT_FILE):
    m = Measurements()
    m.load(measurement_box_file)
    logger.debug('Measurement metos box dict loaded from {}'.format(measurement_box_file))
    return m
