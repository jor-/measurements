import pathlib

import numpy as np
import scipy.sparse

import overrides

import util.cache
import util.plot.save
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

    def _value_function(self, plot_type):
        if plot_type == 'means':
            value_function = np.mean
        elif plot_type == 'inter_quartile_ranges':
            value_function = scipy.stats.iqr
        elif plot_type == 'standard_deviations':
            value_function = lambda x: np.std(x, ddof=1)
        elif plot_type == 'variances':
            value_function = lambda x: np.var(x, ddof=1)
        else:
            raise ValueError(f'Plot type {plot_type} is unknown.')
        return value_function

    def plot_correlation(self, axis, file, plot_type='means', use_abs=True, use_sample_correlation=False, overwrite=False):
        if overwrite or not pathlib.Path(file).exists():
            axis = self._prepare_axis(axis)
            if len(axis) != 1:
                raise ValueError(f'The parameter axis has to be an integer value but it is {axis}.')

            correlation = self.correlation_array(axis=axis, use_sample_correlation=use_sample_correlation)
            assert correlation.shape[1] == 3

            value_function = self._value_function(plot_type)
            util.plot.save.imshow_dataset_values(file, correlation, value_function, use_abs=use_abs, overwrite=overwrite)
        return file

    def plot_autocorrelation(self, axis, file, plot_type='means', use_abs=True, use_sample_correlation=False, overwrite=False):
        if overwrite or not pathlib.Path(file).exists():
            axis = self._prepare_axis(axis)
            if len(axis) > 2:
                raise ValueError(f'The parameter axis has to be one or two integer values but it is {axis}.')

            autocorrelation = self.autocorrelation_array(axis=axis, use_sample_correlation=use_sample_correlation)
            assert autocorrelation.shape[1] == len(axis) + 1

            value_function = self._value_function(plot_type)
            util.plot.save.scatter_dataset_values(file, autocorrelation, value_function, use_abs=use_abs, overwrite=overwrite)
        return file

    def plot_autocorrelation_violins(self, axis, file, use_sample_correlation=False, overwrite=False):
        if overwrite or not pathlib.Path(file).exists():
            axis = self._prepare_axis(axis)
            if len(axis) != 1:
                raise ValueError(f'The parameter axis has to be an integer value but it is {axis}.')

            autocorrelation = self.autocorrelation_array(axis=axis, use_sample_correlation=use_sample_correlation)
            assert autocorrelation.shape[1] == 2

            x = autocorrelation[:, 0]
            y = autocorrelation[:, 1]
            positions = np.unique(x)
            dataset = tuple(np.sort(y[x == p]) for p in positions)

            util.plot.save.violin(file, positions, dataset, overwrite=overwrite)
        return file


class CorrelationCache(Correlation):

    # *** array files *** #

    def _format_filename(self, base_file, axis, use_sample_correlation=False):
        axis = self._prepare_axis(axis)
        axis_str = ','.join(map(str, axis))

        m = self.measurements_object

        if use_sample_correlation:
            file = base_file.format(
                tracer=m.tracer,
                data_set=m.data_set_name,
                sample_correlation_id=m.sample_correlation_id,
                dtype=m.dtype_correlation,
                axis=axis_str)
        else:
            file = base_file.format(
                tracer=m.tracer,
                data_set=m.data_set_name,
                correlation_id=m.correlation_id,
                dtype=m.dtype_correlation,
                axis=axis_str)

        return file

    @util.cache.file.decorator()
    @overrides.overrides
    def correlation_array(self, axis=None, use_sample_correlation=False):
        return super().correlation_array(axis=axis, use_sample_correlation=use_sample_correlation)

    def correlation_array_cache_file(self, axis=None, use_sample_correlation=False):
        if use_sample_correlation:
            base_file = measurements.universal.constants.CORRELATION_ARRAY_SAMPLE_CORRELATION_MATRIX_FILE
        else:
            base_file = measurements.universal.constants.CORRELATION_ARRAY_CORRELATION_MATRIX_FILE
        return self._format_filename(base_file, axis, use_sample_correlation=use_sample_correlation)

    @util.cache.file.decorator()
    @overrides.overrides
    def autocorrelation_array(self, axis=None, use_sample_correlation=False):
        return super().autocorrelation_array(axis=axis, use_sample_correlation=use_sample_correlation)

    def autocorrelation_array_cache_file(self, axis=None, use_sample_correlation=False):
        if use_sample_correlation:
            base_file = measurements.universal.constants.AUTOCORRELATION_ARRAY_SAMPLE_CORRELATION_MATRIX_FILE
        else:
            base_file = measurements.universal.constants.AUTOCORRELATION_ARRAY_CORRELATION_MATRIX_FILE
        return self._format_filename(base_file, axis, use_sample_correlation=use_sample_correlation)

    # *** plot files *** #

    def _plot_format_filename(self, folder_name, plot_name, axis, use_sample_correlation=False):
        axis = self._prepare_axis(axis)
        axis_str = ','.join(map(str, axis))

        m = self.measurements_object

        if use_sample_correlation:
            kind_id = m.sample_correlation_id
            kind_folder_name = 'sample_correlation'
        else:
            kind_id = m.correlation_id
            kind_folder_name = 'correlation'

        plot_name = plot_name.format(axis=axis_str)

        file = measurements.plot.constants.PLOT_FILE.format(
            tracer=m.tracer,
            data_set=m.data_set_name,
            kind=str(pathlib.PurePath('correlation', kind_folder_name, folder_name)),
            kind_id=kind_id,
            plot_name=plot_name,
            file_extension=measurements.plot.constants.PLOT_DEFAULT_FILE_EXTENSION)

        return file

    @overrides.overrides
    def plot_correlation(self, axis, file=None, plot_type='means', use_abs=True, use_sample_correlation=False, overwrite=False):
        if file is None:
            folder_name = f'correlation_{plot_type}'
            plot_name = folder_name + '_-_axis_{axis}' + f'_-_abs_{use_abs}'
            file = self._plot_format_filename(folder_name, plot_name, axis, use_sample_correlation=use_sample_correlation)
        super().plot_correlation(axis, file, plot_type=plot_type, use_abs=use_abs, use_sample_correlation=use_sample_correlation, overwrite=overwrite)
        return file

    @overrides.overrides
    def plot_autocorrelation(self, axis, file=None, plot_type='means', use_abs=True, use_sample_correlation=False, overwrite=False):
        if file is None:
            folder_name = f'auto_correlation_{plot_type}'
            plot_name = folder_name + '_-_axis_{axis}' + f'_-_abs_{use_abs}'
            file = self._plot_format_filename(folder_name, plot_name, axis, use_sample_correlation=use_sample_correlation)
        super().plot_autocorrelation(axis, file, plot_type=plot_type, use_abs=use_abs, use_sample_correlation=use_sample_correlation, overwrite=overwrite)
        return file

    @overrides.overrides
    def plot_autocorrelation_violins(self, axis, file=None, use_sample_correlation=False, overwrite=False):
        if file is None:
            folder_name = 'correlation_violins'
            plot_name = folder_name + '_-_axis_{axis}'
            file = self._plot_format_filename(folder_name, plot_name, axis, use_sample_correlation=use_sample_correlation)
        super().plot_autocorrelation_violins(axis, file, use_sample_correlation=use_sample_correlation, overwrite=overwrite)
        return file
