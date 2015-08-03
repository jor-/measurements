import os.path

from ..constants import ANALYSIS_DIR

VALUE_DIR = os.path.join(ANALYSIS_DIR, 'correlation')
MEASUREMENTS_SAME_POINTS_FILENAME = 'measurements_categorized.same_point_measurements.min_{min_values:0>2}_measurements.ppy'
VALUES_MEASUREMENTS_FILENAME = 'measurements_categorized.{type}_nonstationary.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.ppy'
VALUES_MEASUREMENTS_TRANSFORMED_FILENAME = 'measurements_categorized.{type}_nonstationary.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.transformed.ppy'



## for uncategorized correlation

import numpy as np

import measurements.po4.wod.constants

T_DIM = measurements.po4.wod.constants.SAMPLE_LSM.t_dim
Z = measurements.po4.wod.constants.SAMPLE_LSM.z
B = ((Z[1:] - Z[:-1]) / 2)
EQUAL_BOUNDS = [1./T_DIM, 1, 1, np.array([Z[:-1], B]).T]
