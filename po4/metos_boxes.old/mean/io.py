import numpy as np

import measurements.land_sea_mask.data
import measurements.po4.metos_boxes.util.interpolate
import measurements.util.data
import measurements.util.map

from measurements.po4.metos_boxes.data.constants import METOS_BOXES_DICT_FILE
from measurements.po4.metos_boxes.mean.constants import METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE, METOS_BOXES_MEANS_INTERPOLATED_MAP_FILE, INTERPOLATOR_SETUP

import util.logging
logger = util.logging.get_logger()


def calculate_interpolated_list(time_dim, min_measurements=1, interpolator_setup=INTERPOLATOR_SETUP, parallel=True):
    logger.debug('Calculating po4 mean for ndop boxes in list shape with time dim {} and min measurements {}.'.format(time_dim, min_measurements))
    
    ## get means
    m = measurements.util.data.Measurements_Unsorted()
    m.load(METOS_BOXES_DICT_FILE)
    m.discard_year()
    m.categorize_indices((1./time_dim, None, None, None))
    
    data = m.means(minimum_measurements=min_measurements)
    data_points = data[:, :-1]
    data_values = data[:, -1]
    
    ## interpolate all boxes
    (interpolated_points, interpolated_values) = measurements.po4.metos_boxes.util.interpolate.metos_boxes(data_points, data_values, time_dim, interpolator_setup=interpolator_setup, parallel=parallel)
    interpolated_data = np.concatenate((interpolated_points, interpolated_values[:,np.newaxis]), axis=1)
    
    return interpolated_data


def get_file(file, time_dim, min_measurements=1, interpolator_setup=INTERPOLATOR_SETUP):
    return file.format(time_dim=time_dim, min_measurements=min_measurements, interpolator_setup=str(interpolator_setup).replace(' ',''))
    

def save_interpolated_list(time_dim, min_measurements=1, interpolator_setup=INTERPOLATOR_SETUP, parallel=True):
    interpolated_list_file = get_file(METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE, time_dim, min_measurements, interpolator_setup)
    interpolated_data = calculate_interpolated_list(time_dim, min_measurements=min_measurements, interpolator_setup=INTERPOLATOR_SETUP, parallel=parallel)
    np.save(interpolated_list_file, interpolated_data)
    logger.debug('Interpolated mean of po4 saved in list shape to {}.'.format(interpolated_list_file))


def load_interpolated_list(time_dim, min_measurements=1, interpolator_setup=INTERPOLATOR_SETUP):
    interpolated_list_file = get_file(METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE, time_dim, min_measurements, interpolator_setup)
    deviation = np.load(interpolated_list_file)
    logger.debug('Loaded po4 mean for ndop boxes in list shape from {}.'.format(interpolated_list_file))
    return deviation



def calculate_interpolated_map(time_dim, min_measurements=1, interpolator_setup=INTERPOLATOR_SETUP):
    interpolated_list = load_interpolated_list(time_dim, min_measurements=min_measurements, interpolator_setup=interpolator_setup)
    lsm = measurements.land_sea_mask.data.LandSeaMaskTMM()
    interpolated_map = measurements.util.map.insert_values_in_map(interpolated_list, lsm, no_data_value=np.inf)
    return interpolated_map


def save_interpolated_map(time_dim, min_measurements=1, interpolator_setup=INTERPOLATOR_SETUP):
    interpolated_map_file = get_file(METOS_BOXES_MEANS_INTERPOLATED_MAP_FILE, time_dim, min_measurements, interpolator_setup)
    interpolated_data = calculate_interpolated_map(time_dim, min_measurements=min_measurements, interpolator_setup=interpolator_setup)
    np.save(interpolated_map_file, interpolated_data)
    logger.debug('Interpolated mean of po4 saved to {} in map shape.'.format(interpolated_map_file))


def load_interpolated_map(time_dim, min_measurements=1, interpolator_setup=INTERPOLATOR_SETUP):
    interpolated_map_file =get_file(METOS_BOXES_MEANS_INTERPOLATED_MAP_FILE, time_dim, min_measurements, interpolator_setup)
    interpolated_data = np.load(interpolated_map_file)
    logger.debug('Loaded po4 mean for ndop boxes in map shape from {}.'.format(interpolated_map_file))
    return interpolated_data

    