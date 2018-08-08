import pathlib

import numpy as np
import scipy.sparse

import overrides

import util.cache
import util.plot
import measurements.universal.constants


class Autocorrelation():

    def __init__(self, measurements_object):
        self.measurements_object = measurements_object

    def _prepare_axis(self, axis):
        if axis is None:
            m = self.measurements_object.points.shape[1]
            axis = np.arange(m)
        else:
            axis = np.asarray(axis, dtype=int)
            if axis.ndim == 0:
                axis = axis.reshape(-1)
        assert axis.ndim == 1
        return axis

    def autocorrelation(self, axis=None, use_sample_correlation=False):
        if axis is None:
            # get strict lower triangle of correlation matrix
            mo = self.measurements_object
            if use_sample_correlation:
                A = mo.correlations_own_sample_matrix
            else:
                A = mo.correlations_own
            A = scipy.sparse.tril(A, k=-1)

            if A.format not in ('csc', 'csr'):
                A = A.tocsc(copy=False)

            # get points
            points = mo.points
            points = mo.sample_lsm.coordinates_to_map_indices(points, discard_year=False, int_indices=True)
            points = mo.sample_lsm.map_indices_to_coordinates(points)

            # calculate autocorrelation
            autocorrelation_array_len = A.nnz
            autocorrelation_array = np.empty((autocorrelation_array_len, points.shape[1] + 1))

            n = 0
            m = 0
            for i in range(points.shape[0]):
                point = points[i]
                correlations = A.data[A.indptr[i]:A.indptr[i + 1]]
                correlated_points_indices = A.indices[A.indptr[i]:A.indptr[i + 1]]
                correlated_points = points[correlated_points_indices]
                points_diff = np.abs(point - correlated_points)
                m += len(correlations)
                autocorrelation_array[n:m, :-1] = points_diff
                autocorrelation_array[n:m, -1] = correlations
                n = m
            assert n == autocorrelation_array_len

        else:
            autocorrelation_array = self.autocorrelation(axis=None, use_sample_correlation=use_sample_correlation)
            # remove unwanted axes
            axis = self._prepare_axis(axis)
            for i in range(autocorrelation_array.shape[1] - 1):
                if i not in axis:
                    autocorrelation_array = autocorrelation_array[autocorrelation_array[:, i] == 0]
            axis = axis.tolist()
            axis.append(-1)
            autocorrelation_array = autocorrelation_array[:, axis]

        # return
        return autocorrelation_array

    def plot(self, axis, file, use_sample_correlation=False):
        axis = self._prepare_axis(axis)
        if len(axis) != 1:
            raise ValueError(f'axis has to be an integer but it is {axis}.')
        autocorrelation = self.autocorrelation(axis=axis, use_sample_correlation=use_sample_correlation)
        assert autocorrelation.shape[1] == 2
        x = autocorrelation[:, 0]
        reliable_decimals = np.finfo(x.dtype).precision - 4
        x = np.around(x, decimals=reliable_decimals)
        y = autocorrelation[:, 1]
        positions = np.unique(x)
        dataset = tuple(np.sort(y[x == p]) for p in positions)
        util.plot.violin(positions, dataset, file)
        return file


class AutocorrelationCache(Autocorrelation):

    @util.cache.file.decorator()
    @overrides.overrides
    def autocorrelation(self, axis=None, use_sample_correlation=False):
        return super().autocorrelation(axis=axis, use_sample_correlation=use_sample_correlation)

    def autocorrelation_cache_file(self, axis=None, use_sample_correlation=False):
        axis = self._prepare_axis(axis)
        axis_str = ','.join(map(str, axis))

        m = self.measurements_object

        if use_sample_correlation:
            return measurements.universal.constants.AUTOCORRELATION_SAMPLE_CORRELATION_MATRIX_FILE.format(
                tracer=m.tracer,
                data_set=m.data_set_name,
                sample_lsm=m.sample_lsm,
                min_measurements_correlation=m.min_measurements_correlation,
                min_abs_correlation=m.min_abs_correlation,
                max_abs_correlation=m.max_abs_correlation,
                standard_deviation_id=m.standard_deviation_id_without_sample_lsm,
                dtype=m.dtype_correlation,
                axis=axis_str)
        else:
            return measurements.universal.constants.AUTOCORRELATION_CORRELATION_MATRIX_FILE.format(
                tracer=m.tracer,
                data_set=m.data_set_name,
                sample_lsm=m.sample_lsm,
                min_measurements_correlation=m.min_measurements_correlation,
                min_abs_correlation=m.min_abs_correlation,
                max_abs_correlation=m.max_abs_correlation,
                decomposition_type=m.decomposition_type_correlations,
                permutation_method_decomposition_correlation=m.permutation_method_decomposition_correlation,
                decomposition_min_diag_value=m.min_diag_value_decomposition_correlation,
                standard_deviation_id=m.standard_deviation_id_without_sample_lsm,
                dtype=m.dtype_correlation,
                axis=axis_str)

    @overrides.overrides
    def plot(self, axis, file=None, use_sample_correlation=False):
        if file is None:
            file = self.plot_file(axis=axis, use_sample_correlation=use_sample_correlation)
        return super().plot(axis, file, use_sample_correlation=use_sample_correlation)

    def plot_file(self, axis=None, use_sample_correlation=False):
        autocorrelation_cache_file = self.autocorrelation_cache_file(axis=axis, use_sample_correlation=use_sample_correlation)
        autocorrelation_cache_file = pathlib.PurePath(autocorrelation_cache_file)
        plot_file = autocorrelation_cache_file.with_suffix('.svg')
        return plot_file
