import os.path

from ..constants import ANALYSIS_DIR

VALUE_DIR = os.path.join(ANALYSIS_DIR, 'correlation')
MEASUREMENTS_SAME_POINTS_FILENAME = 'measurements_categorized.same_point_measurements.min_{min_values:0>2}_measurements.ppy'
SAMPLE_VALUE_DICT_FILENAME = 'measurements_categorized.{type}_nonstationary.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.ppy'
SAMPLE_VALUE_DICT_TRANSFORMED_FILENAME = 'measurements_categorized.{type}_nonstationary.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.transformed.ppy'
SAME_BOX_SAMPLE_CORRELATION_DICT_FILENAME = 'measurements_categorized.same_box_correlation.ppy'
SAME_BOX_SAMPLE_CORRELATION_DICT_TRANSFORMED_FILENAME = 'measurements_categorized.same_box_correlation.transformed.ppy'
DIFFERENT_BOXES_SAMPLE_CORRELATION_DICT_FILENAME = 'measurements_categorized.different_boxes_correlation.nonstationary.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.ppy'
DIFFERENT_BOXES_SAMPLE_CORRELATION_DICT_TRANSFORMED_FILENAME = 'measurements_categorized.different_boxes_correlation.nonstationary.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.transformed.ppy'



## for uncategorized correlation

import numpy as np

import measurements.po4.wod.constants

T_DIM = measurements.po4.wod.constants.SAMPLE_LSM.t_dim
Z = measurements.po4.wod.constants.SAMPLE_LSM.z
B = ((Z[1:] - Z[:-1]) / 2)
EQUAL_BOUNDS = [1./T_DIM, 1, 1, np.array([Z[:-1], B]).T]
