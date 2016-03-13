import os.path

from ..constants import ANALYSIS_DIR

VALUE_DIR = os.path.join(ANALYSIS_DIR, 'correlation')

MEASUREMENTS_SAME_POINTS_FILENAME = 'measurements.same_point_measurements.categorized_{sample_lsm}.min_{min_values:0>2}_measurements.ppy'

DIFFERENT_BOXES_SAMPLE_VALUE_DICT_FILENAME = 'different_boxes_{type}_dict.nonstationary.categorized_{sample_lsm}.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.ppy'
DIFFERENT_BOXES_SAMPLE_VALUE_DICT_TRANSFORMED_FILENAME = 'different_boxes_{type}_dict.nonstationary.categorized_{sample_lsm}.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.transformed.ppy'

SAME_BOX_SAMPLE_COVARIANCE_DICT_FILENAME = 'same_box_covariance_dict.categorized_{sample_lsm}.ppy'
SAME_BOX_SAMPLE_COVARIANCE_DICT_TRANSFORMED_FILENAME = 'same_box_covariance_dict.categorized_{sample_lsm}.transformed.ppy'



## for uncategorized correlation

import numpy as np

import measurements.po4.wod.constants

T_DIM = measurements.po4.wod.constants.SAMPLE_LSM.t_dim
Z = measurements.po4.wod.constants.SAMPLE_LSM.z
B = ((Z[1:] - Z[:-1]) / 2)
EQUAL_BOUNDS = [1./T_DIM, 1, 1, np.array([Z[:-1], B]).T]
