import os.path

from measurements.all.constants import BASE_DIR

BASE_DIR = os.path.join(BASE_DIR, 'pw')
CORRELATION_DIR = os.path.join(BASE_DIR, 'correlation')
# MATRIX_FILENAME_PREFIX = 'matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.positive_definite.'
# MATRIX_FILENAME =  'matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.positive_definite.{type}.ppy'
# DOK_MATRIX_FILENAME =  MATRIX_FILENAME_PREFIX + 'dok.ppy'
# CSR_MATRIX_FILENAME =  MATRIX_FILENAME_PREFIX + 'csr.ppy'
# CHOLESKY_L_MATRIX_FILENAME =  MATRIX_FILENAME_PREFIX + 'L.csc.ppy'
# CHOLESKY_P_MATRIX_FILENAME =  MATRIX_FILENAME_PREFIX + 'P.csc.ppy'


POINT_INDEX_FILENAME = 'point_index_{}_dict.ppy'
CORRELATION_MATRIX_SAME_BOX_FILENAME = 'correlation_matrix.same_boxes.lower_triangle.{type}.ppy'
SAMPLE_CORRELATION_AND_QUANTITY_MATRIX_FILENAME = 'sample_correlation_and_quantity_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.{type}.ppy'
CORRELATION_MATRIX_FILENAME = 'correlation_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.{type}.ppy'
CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME = 'correlation_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.positive_definite.{ordering_method}_ordering.reordering_{reordering}.{type}.ppy'
CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME = 'correlation_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.positive_definite.{ordering_method}_ordering.reordering_{reordering}.reduction_factors.npy'
CORRELATION_MATRIX_CHOLESKY_FACTORS_FILENAME = 'correlation_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.positive_definite.{ordering_method}_ordering.reordering_{reordering}.cholesky_factors.{type}.ppy'