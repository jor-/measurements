import os.path

import measurements.all.pw.correlation

import util.plot


def correlation_matrix_sparse_pattern(min_values, max_year_diff=float('inf'), path='/tmp/', dpis=(200, 400, 800, 2000)):
    markersize = 1

    cm = measurements.all.pw.correlation.CorrelationMatrix(min_values, max_year_diff=max_year_diff)

    C = cm.correlation_matrix
    # P, L = cm.correlation_matrix_cholesky_decomposition
    # L = L.tocoo()
    # C_reordered = (P*C*P.transpose()).tocoo()
    C = C.tocoo()

    for dpi in dpis:
        file = os.path.join(path, 'correlation_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.{dpi:0>4}_dpi.png'.format(min_values=min_values, max_year_diff=max_year_diff, dpi=dpi))
        util.plot.spy(C, file, markersize=markersize, dpi=dpi)

        # file = os.path.join(path, 'correlation_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.cholesky_L.{dpi:0>4}_dpi.png'.format(min_values=min_values, max_year_diff=max_year_diff, dpi=dpi))
        # util.plot.spy(L, file, markersize=markersize, dpi=dpi)

   #    #   file = os.path.join(path, 'correlation_matrix.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.cholesky_reordered.{dpi:0>4}_dpi.png'.format(min_values=min_values, max_year_diff=max_year_diff, dpi=dpi))
        # util.plot.spy(C_reordered, file, markersize=markersize, dpi=dpi)

