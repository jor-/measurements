import os.path

import measurements.all.pw.correlation

import util.plot


def correlation_matrix_sparse_pattern(min_values, max_year_diff=float('inf'), axis_labels=False, path='/tmp/', dpis=(200, 400, 800, 2000)):
    markersize = 1

    cm = measurements.all.pw.correlation.CorrelationMatrix(min_values, max_year_diff=max_year_diff)

    C = cm.correlation_matrix()
    C = C.tocoo()
    
    for dpi in dpis:
        file = os.path.join(path, 'correlation_matrix.categorized_lsm_12_woa13r.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_0.01.max_abs_correlation_0.99.{dpi:0>4}_dpi.png'.format(min_values=min_values, max_year_diff=max_year_diff, dpi=dpi))
        util.plot.spy(C, file, axis_labels=axis_labels, markersize=markersize, dpi=dpi)


def different_boxes_correlation_lower_triangle_matrix_sparse_pattern(min_values, max_year_diff=float('inf'), axis_labels=False, path='/tmp/', dpis=(200, 400, 800, 2000)):
    markersize = 1

    cm = measurements.all.pw.correlation.CorrelationMatrix(min_values, max_year_diff=max_year_diff)

    C = cm.different_boxes_correlation_lower_triangle_matrix()
    C = C.tocoo()

    for dpi in dpis:
        file = os.path.join(path, 'different_boxes_correlation_lower_triangle_matrix.categorized_lsm_12_woa13r.min_{min_values:0>2}_measurements.max_{max_year_diff:0>2}_year_diff.min_abs_correlation_0.01.max_abs_correlation_0.99.{dpi:0>4}_dpi.png'.format(min_values=min_values, max_year_diff=max_year_diff, dpi=dpi))
        util.plot.spy(C, file, axis_labels=axis_labels, markersize=markersize, dpi=dpi)