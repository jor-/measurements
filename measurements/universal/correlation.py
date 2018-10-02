import pathlib

import numpy as np
import scipy.sparse

import overrides

import util.cache
import util.plot
import measurements.universal.constants


class Correlation():

    def __init__(self, measurements_object):
        self.measurements_object = measurements_object

    def _prepare_axis(self, axis):
        if axis is None:
            point_dim = self.measurements_object.points.shape[1]
            axis = np.arange(point_dim)
        else:
            axis = np.asarray(axis, dtype=int)
            if axis.ndim == 0:
                axis = axis.reshape(-1)
        assert axis.ndim == 1
        return axis

    def correlation_array(self, axis=None, use_sample_correlation=False):
        mo = self.measurements_object

        if axis is None:
            # get strict lower triangle of correlation matrix
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
            correlation_array_len = A.nnz
            point_dim = points.shape[1]
            correlation_array = np.empty((correlation_array_len, point_dim * 2 + 1))

            n = 0
            m = 0
            for i in range(points.shape[0]):
                point = points[i]
                correlations = A.data[A.indptr[i]:A.indptr[i + 1]]
                correlated_points_indices = A.indices[A.indptr[i]:A.indptr[i + 1]]
                correlated_points = points[correlated_points_indices]
                m += len(correlations)
                correlation_array[n:m, :point_dim] = point
                correlation_array[n:m, point_dim:-1] = correlated_points
                correlation_array[n:m, -1] = correlations
                n = m
            assert n == correlation_array_len

        else:
            correlation_array = self.correlation_array(axis=None, use_sample_correlation=use_sample_correlation)
            point_dim = (correlation_array.shape[1] - 1) / 2
            assert point_dim.is_integer()
            point_dim = int(point_dim)

            # remove unwanted axes
            axis = self._prepare_axis(axis)
            for i in range(point_dim):
                if i not in axis:
                    mask = (mo.sample_lsm.coordinates_to_map_indices_single_axis(correlation_array[:, i], i, discard_year=False, int_indices=True) ==
                            mo.sample_lsm.coordinates_to_map_indices_single_axis(correlation_array[:, i + point_dim], i, discard_year=False, int_indices=True))
                    correlation_array = correlation_array[mask]
            correlation_array_axes = axis.tolist() + (axis + point_dim).tolist() + [-1]
            correlation_array = correlation_array[:, correlation_array_axes]
            assert correlation_array.shape[1] == len(axis) * 2 + 1

        # return
        return correlation_array

    def autocorrelation_array(self, axis=None, use_sample_correlation=False):
        correlation_array = self.correlation_array(axis=axis, use_sample_correlation=use_sample_correlation)
        # calculate dim of points
        assert correlation_array.ndim == 2
        (m, n) = correlation_array.shape
        n = (n - 1) / 2
        assert n.is_integer()
        n = int(n)
        m = 2 * n
        # calculate points difference
        reliable_decimals = np.finfo(correlation_array.dtype).precision - 4
        correlation_array[:, n:m] = correlation_array[:, :n] - correlation_array[:, n:m]
        correlation_array[:, n:m] = np.abs(correlation_array[:, n:m])
        correlation_array[:, n:m] = np.around(correlation_array[:, n:m], decimals=reliable_decimals)
        # return autocorrelation array
        autocorrelation_array = correlation_array[:, n:]
        return autocorrelation_array

    def plot_autocorrelation(self, axis, file, use_sample_correlation=False, mean_only=True):
        axis = self._prepare_axis(axis)
        autocorrelation = self.autocorrelation_array(axis=axis, use_sample_correlation=use_sample_correlation)
        assert autocorrelation.shape[1] == len(axis) + 1
        x = autocorrelation[:, :-1]
        x = np.linalg.norm(x, ord=2, axis=1)
        y = autocorrelation[:, -1]
        positions = np.unique(x)
        if mean_only:
            dataset = np.array(tuple(np.mean(np.abs((y[x == p]))) for p in positions))
            util.plot.scatter(file, positions, dataset)
        else:
            dataset = tuple(np.sort(y[x == p]) for p in positions)
            util.plot.violin(file, positions, dataset)
        return file


class CorrelationCache(Correlation):

    def _format_file_sample_correlation(self, file, axis):
        axis = self._prepare_axis(axis)
        axis_str = ','.join(map(str, axis))

        m = self.measurements_object

        file = file.format(
            tracer=m.tracer,
            data_set=m.data_set_name,
            sample_lsm=m.sample_lsm,
            min_measurements_correlation=m.min_measurements_correlation,
            min_abs_correlation=m.min_abs_correlation,
            max_abs_correlation=m.max_abs_correlation,
            standard_deviation_id=m.standard_deviation_id_without_sample_lsm,
            dtype=m.dtype_correlation,
            axis=axis_str)

        return file

    def _format_file_correlation(self, file, axis):
        axis = self._prepare_axis(axis)
        axis_str = ','.join(map(str, axis))

        m = self.measurements_object

        file = file.format(
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

        return file

    @util.cache.file.decorator()
    @overrides.overrides
    def correlation_array(self, axis=None, use_sample_correlation=False):
        return super().correlation_array(axis=axis, use_sample_correlation=use_sample_correlation)

    def correlation_array_cache_file(self, axis=None, use_sample_correlation=False):
        if use_sample_correlation:
            return self._format_file_sample_correlation(measurements.universal.constants.CORRELATION_ARRAY_SAMPLE_CORRELATION_MATRIX_FILE, axis)
        else:
            return self._format_file_correlation(measurements.universal.constants.CORRELATION_ARRAY_CORRELATION_MATRIX_FILE, axis)

    @util.cache.file.decorator()
    @overrides.overrides
    def autocorrelation_array(self, axis=None, use_sample_correlation=False):
        return super().autocorrelation_array(axis=axis, use_sample_correlation=use_sample_correlation)

    def autocorrelation_array_cache_file(self, axis=None, use_sample_correlation=False):
        if use_sample_correlation:
            return self._format_file_sample_correlation(measurements.universal.constants.AUTOCORRELATION_ARRAY_SAMPLE_CORRELATION_MATRIX_FILE, axis)
        else:
            return self._format_file_correlation(measurements.universal.constants.AUTOCORRELATION_ARRAY_CORRELATION_MATRIX_FILE, axis)

    @overrides.overrides
    def plot_autocorrelation(self, axis, file=None, use_sample_correlation=False, mean_only=True):
        if file is None:
            file = self.plot_autocorrelation_file(axis=axis,
                                                  use_sample_correlation=use_sample_correlation,
                                                  mean_only=mean_only)
        if not pathlib.Path(file).exists():
            super().plot_autocorrelation(axis, file,
                                         use_sample_correlation=use_sample_correlation,
                                         mean_only=mean_only)
        return file

    def plot_autocorrelation_file(self, axis=None, use_sample_correlation=False, mean_only=True):
        autocorrelation_cache_file = self.autocorrelation_array_cache_file(axis=axis, use_sample_correlation=use_sample_correlation)
        plot_file = autocorrelation_cache_file
        if mean_only:
            plot_file = pathlib.PurePath(plot_file)
            plot_file = plot_file.parent.joinpath('average_' + plot_file.name)
        plot_file = measurements.universal.constants.plot_file(plot_file)
        return plot_file
