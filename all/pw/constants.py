import os.path

from measurements.all.constants import BASE_DIR

BASE_DIR = os.path.join(BASE_DIR, 'pw')
CORRELATION_DIR = os.path.join(BASE_DIR, 'correlation')

POINT_INDEX_FILENAME = 'point_index_{year_type}_dict.categorized_{sample_lsm}.ppy'


SAME_BOX_CORRELATION_LOWER_TRIANGLE_MATRIX_FILENAME = 'same_box_correlation_lower_triangle_matrix.categorized_{sample_lsm}.min_abs_correlation_{min_abs_correlation}.{type}.ppy'

DIFFERENT_BOXES_CORRELATION_LOWER_TRIANGLE_MATRIX_FILENAME = 'different_boxes_correlation_lower_triangle_matrix.categorized_{sample_lsm}.min_{min_measurements:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_{min_abs_correlation}.{type}.ppy'

DIFFERENT_BOXES_QUANTITY_LOWER_TRIANGLE_MATRIX_FILENAME = 'different_boxes_quantity_lower_triangle_matrix.categorized_{sample_lsm}.min_{min_measurements:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_{min_abs_correlation}.{type}.ppy'

CORRELATION_MATRIX_FILENAME = 'correlation_matrix.categorized_{sample_lsm}.min_{min_measurements:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_{min_abs_correlation}.max_abs_correlation_{max_abs_correlation}.{type}.ppy'


CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME = 'positive_definite.correlation_matrix.categorized_{sample_lsm}.min_{min_measurements:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_{min_abs_correlation}.max_abs_correlation_{max_abs_correlation}.{ordering_method}_ordering.reordering_{reordering}.min_diag_{min_diag_value:.0e}.{type}.ppy'
CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME = 'positive_definite.correlation_matrix.categorized_{sample_lsm}.min_{min_measurements:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_{min_abs_correlation}.max_abs_correlation_{max_abs_correlation}.{ordering_method}_ordering.reordering_{reordering}.min_diag_{min_diag_value:.0e}.reduction_factors.npy'
CORRELATION_MATRIX_CHOLESKY_FACTORS_FILENAME = 'positive_definite.correlation_matrix.categorized_{sample_lsm}.min_{min_measurements:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_{min_abs_correlation}.max_abs_correlation_{max_abs_correlation}.{ordering_method}_ordering.reordering_{reordering}.min_diag_{min_diag_value:.0e}.cholesky_factors.{type}.ppy'


from measurements.po4.wod.constants import SAMPLE_LSM
from measurements.constants import CORRELATION_MIN_DIAG_VALUE_POSITIVE_DEFINITE_APPROXIMATION