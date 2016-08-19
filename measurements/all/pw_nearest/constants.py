import os.path

from measurements.all.constants import BASE_DIR

BASE_DIR = os.path.join(BASE_DIR, 'pw_nearest_{lsm}_{max_land_boxes}')
CORRELATION_DIR = os.path.join(BASE_DIR, 'correlation')

from measurements.all.pw.constants import CORRELATION_MATRIX_FILENAME, CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME, CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME, CORRELATION_MATRIX_CHOLESKY_FACTORS_FILENAME