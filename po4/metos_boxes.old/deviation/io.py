import numpy as np

# import logging
# logger = logging.getLogger(__name__)
# 
# from measurements.po4.wod.deviation.model import Deviation_Model

import measurements.po4.wod.deviation.estimation
import measurements.po4.metos_boxes.util.interpolate

from measurements.po4.metos_boxes.data.constants import METOS_BOXES_DICT_FILE
from ndop.model.constants import METOS_X_DIM
from .constants import T_DIM, METOS_BOXES_DEVIATIONS_INTERPOLATED_LIST_FILE, INTERPOLATOR_SETUP

import measurements.po4.metos_boxes.util.interpolate

import util.logging
logger = util.logging.get_logger()



def calculate_interpolated_deviation_list(t_dim, interpolator_setup=INTERPOLATOR_SETUP, parallel=True):
    logger.debug('Calculating standard deviation for ndop boxes with time dim {}.'.format(t_dim))
    
    ## sample deviation
    deviation_estimation = measurements.po4.wod.deviation.estimation.deviations_from_measurements(measurements_file=METOS_BOXES_DICT_FILE, separation_values=(1./t_dim, None, None, None))
    deviation_estimation_points = deviation_estimation[:, :-1]
    deviation_estimation_values = deviation_estimation[:, -1]
    
    ## interpolate deviation
#     deviation_points, deviation_values = measurements.po4.metos_boxes.util.interpolate.metos_boxes(deviation_estimation_points, deviation_estimation_values, t_dim, amount_of_wrap_around=AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=NUMBER_OF_LINEAR_INTERPOLATOR, total_overlapping_linear_interpolators=TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR, parallel=False)
    deviation_points, deviation_values = measurements.po4.metos_boxes.util.interpolate.metos_boxes(deviation_estimation_points, deviation_estimation_values, t_dim, interpolator_setup=interpolator_setup, parallel=parallel)
    
    ## concatenate
#     deviation_values = deviation_values.reshape([len(deviation_values), 1])
    result = np.concatenate((deviation_points, deviation_values[:, np.newaxis]), axis=1)
    return result


def save_interpolated_deviation_list(t_dim=T_DIM, interpolator_setup=INTERPOLATOR_SETUP, parallel=True):
    
    interpolated_list_file = METOS_BOXES_DEVIATIONS_INTERPOLATED_LIST_FILE.format(t_dim, str(interpolator_setup).replace(' ',''))
    interpolated_data = calculate_interpolated_deviation_list(t_dim, interpolator_setup=interpolator_setup, parallel=parallel)
    np.save(interpolated_list_file, interpolated_data)
    logger.debug('Standard deviation of po4 saved in list shape to {}.'.format(interpolated_list_file))


def load_interpolated_deviation_list(t_dim=T_DIM, interpolator_setup=INTERPOLATOR_SETUP):
    interpolated_list_file = METOS_BOXES_DEVIATIONS_INTERPOLATED_LIST_FILE.format(t_dim, str(interpolator_setup).replace(' ',''))
    interpolated_data = np.load(interpolated_list_file)
    logger.debug('Standard deviation of po4 loaded in list shape from {}.'.format(interpolated_list_file))
    return interpolated_data
