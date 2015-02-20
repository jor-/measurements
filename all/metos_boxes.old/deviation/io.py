import os.path
import numpy as np

import logging
logger = logging.getLogger(__name__)

import measurements.land_sea_mask.data
import measurements.dop.pw.deviation
import measurements.po4.metos_boxes.deviation.io
import measurements.po4.wod.data.results
import measurements.util.map

# from measurements.po4.metos_boxes.deviation.constants import METOS_BOXES_DEVIATIONS_INTERPOLATED_LIST_FILE as METOS_BOXES_PO4_DEVIATIONS_INTERPOLATED_LIST_FILE
# from .constants import T_DIM, METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE
from .constants import METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE, T_DIM, INTERPOLATOR_SETUP
# from measurements.po4.metos_boxes.deviation.constants import T_DIM, INTERPOLATOR_SETUP


def calculate_interpolated_deviation_map(t_dim=T_DIM, interpolator_setup=INTERPOLATOR_SETUP):
    
    ## PO4
#     po4_deviation_list = measurements.po4.metos_boxes.deviation.io.load_interpolated_deviation_list(t_dim=52, interpolated_deviation_list_file=METOS_BOXES_PO4_DEVIATIONS_INTERPOLATED_LIST_FILE)
#     m = measurements.po4.wod.data.results.Measurements_Unsorted()
#     m.add_results(po4_deviation_list[:,:-1], po4_deviation_list[:,-1])
#     m.categorize_indices((1./t_dim,))
#     po4_deviation_map = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
    
    po4_deviation_list = measurements.po4.metos_boxes.deviation.io.load_interpolated_deviation_list(t_dim=t_dim, interpolator_setup=interpolator_setup)
    lsm = measurements.land_sea_mask.data.LandSeaMaskTMM()
    po4_deviation_map = measurements.util.map.insert_values_in_map(po4_deviation_list, lsm, no_data_value=np.inf)
    
    ## DOP
    dop_deviation_map = np.ones_like(po4_deviation_map) * measurements.dop.pw.deviation.get_average_deviation()
    
    ### concatenate
    deviation_map = np.concatenate([dop_deviation_map[np.newaxis, :], po4_deviation_map[np.newaxis, :]], axis=0)
    
    return deviation_map



def save_interpolated_deviation_map(t_dim=T_DIM, interpolator_setup=INTERPOLATOR_SETUP):
    file = METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE.format(t_dim, str(interpolator_setup).replace(' ',''))
    deviation_map = calculate_interpolated_deviation_map(t_dim=t_dim, interpolator_setup=interpolator_setup)
    np.save(file, deviation_map)
    logger.debug('Standard deviation in map shape saved to {}.'.format(file))



def load_interpolated_deviation_map(t_dim=T_DIM, interpolator_setup=INTERPOLATOR_SETUP):
    file = METOS_BOXES_DEVIATIONS_INTERPOLATED_MAP_FILE.format(t_dim, str(interpolator_setup).replace(' ',''))
    deviation_map = np.load(file)
    logger.debug('Standard deviation in map shape saved to {}.'.format(file))
    return deviation_map
