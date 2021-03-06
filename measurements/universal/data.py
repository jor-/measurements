import abc
import os.path

import numpy as np
import scipy.sparse
import overrides

import matrix.approximation.positive_semidefinite
import matrix.decompositions

import measurements.universal.dict
import measurements.universal.interpolate
import measurements.universal.sample_data
import measurements.universal.constants

import util.cache.file
import util.cache.memory
import util.options
import util.str
import util.logging


class Measurements():

    permutation_method_decomposition_correlation = measurements.universal.constants.CORRELATION_DECOMPOSITION_PERMUTATION_METHOD
    decomposition_type_correlations = measurements.universal.constants.CORRELATION_DECOMPOSITION_TYPE
    matrix_format_correlation = measurements.universal.constants.CORRELATION_FORMAT
    dtype_correlation = measurements.universal.constants.CORRELATION_DTYPE

    def __init__(self, tracer=None, data_set_name=None):
        if tracer is not None:
            self._tracer = tracer
        if data_set_name is not None:
            self._data_set_name = data_set_name
        util.logging.debug('{}: initialized with tracer {} and data set {}.'.format(self.__class__.__name__, tracer, data_set_name))

    # *** string representations *** #

    def __repr__(self):
        return '<measurements for tracer "{tracer}" with data set "{data_set_name}">'.format(tracer=self.tracer, data_set_name=self.data_set_name)

    def __str__(self):
        return '{tracer}:{data_set_name}'.format(tracer=self.tracer, data_set_name=self.data_set_name)

    # *** attributes *** #

    @property
    def tracer(self):
        try:
            tracer = self._tracer
        except AttributeError:
            tracer = None
        if tracer is not None:
            return tracer
        else:
            raise ValueError('Tracer is not set.')

    @property
    def data_set_name(self):
        try:
            data_set_name = self._data_set_name
        except AttributeError:
            data_set_name = None
        if data_set_name is not None:
            return data_set_name
        else:
            raise ValueError('Data set name is not set.')

    @property
    @abc.abstractmethod
    def points(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def values(self):
        raise NotImplementedError()

    @property
    def number_of_measurements(self):
        return len(self.points)

    def __len__(self):
        return self.number_of_measurements

    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
    def measurements_dict(self):
        m = measurements.universal.dict.MeasurementsDict()
        m.append_values(self.points, self.values)
        return m

    @property  # TODO no property
    @abc.abstractmethod
    def means(self):
        raise NotImplementedError()

    @property  # TODO no property
    @abc.abstractmethod
    def standard_deviations(self):
        raise NotImplementedError()

    @property  # TODO no property
    def variances(self):
        return self.standard_deviations ** 2

    @property  # TODO no property
    def correlations_own_sample_matrix(self):
        return scipy.sparse.eye(self.number_of_measurements)

    @property  # TODO no property
    def correlations_own_permutation_vector(self):
        if self.permutation_method_decomposition_correlation in matrix.approximation.positive_semidefinite.APPROXIMATION_ONLY_PERMUTATION_METHODS:
            permutation_vector = self.correlations_own_decomposition.p
        else:
            permutation_vector = matrix.permute.permutation_vector(
                self.correlations_own_sample_matrix,
                self.permutation_method_decomposition_correlation)
        return permutation_vector

    @property
    def correlation_decomposition_min_value_D(self):
        try:
            correlation_decomposition_min_value_D = self._correlation_decomposition_min_value_D
        except AttributeError:
            correlation_decomposition_min_value_D = measurements.universal.constants.CORRELATION_DECOMPOSITION_MIN_VALUE_D
        return correlation_decomposition_min_value_D

    @correlation_decomposition_min_value_D.setter
    def correlation_decomposition_min_value_D(self, value):
        if value < 0 or value > 1:
            raise ValueError(f'correlation_decomposition_min_value_D has to be between 0 and 1 but it should be set to {correlation_decomposition_min_value_D}.')
        self._correlation_decomposition_min_value_D = value

    @property
    def correlation_decomposition_min_abs_value_L(self):
        try:
            correlation_decomposition_min_abs_value_L = self._correlation_decomposition_min_abs_value_L
        except AttributeError:
            correlation_decomposition_min_abs_value_L = measurements.universal.constants.CORRELATION_DECOMPOSITION_MIN_ABS_VALUE_L
        return correlation_decomposition_min_abs_value_L

    @correlation_decomposition_min_abs_value_L.setter
    def correlation_decomposition_min_abs_value_L(self, value):
        if value < 0 or value > 1:
            raise ValueError(f'correlation_decomposition_min_abs_value_L has to be between 0 and 1 but it should be set to {correlation_decomposition_min_abs_value_L}.')
        self._correlation_decomposition_min_abs_value_L = value

    @property  # TODO no property
    def correlations_own_decomposition(self):
        if self.permutation_method_decomposition_correlation in matrix.approximation.positive_semidefinite.APPROXIMATION_ONLY_PERMUTATION_METHODS:
            permutation = self.permutation_method_decomposition_correlation
        else:
            permutation = self.correlations_own_permutation_vector
        correlation_matrix_decomposition = matrix.approximation.positive_semidefinite.decomposition(
            self.correlations_own_sample_matrix,
            min_diag_B=1, max_diag_B=1,
            min_diag_D=self.correlation_decomposition_min_value_D,
            min_abs_value_L=self.correlation_decomposition_min_abs_value_L,
            permutation=permutation,
            return_type=self.decomposition_type_correlations)
        return correlation_matrix_decomposition

    @property  # TODO no property
    def correlations_own(self):
        correlation_matrix_decomposition = self.correlations_own_decomposition
        correlation_matrix = correlation_matrix_decomposition.composed_matrix
        correlation_matrix = correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)
        return correlation_matrix

    def correlations_other(self, measurements=None):
        return scipy.sparse.dia_matrix((self.number_of_measurements, measurements.number_of_measurements))

    def correlations(self, measurements=None):
        if measurements is None or measurements == self:
            return self.correlations_own
        else:
            return self.correlations_other(measurements=measurements)


class MeasurementsAnnualPeriodicBase(Measurements):

    sample_lsm = measurements.universal.constants.SAMPLE_LSM
    min_measurements_mean = measurements.universal.constants.MEAN_MIN_MEASUREMENTS
    min_measurements_quantile = measurements.universal.constants.QUANTILE_MIN_MEASUREMENTS
    min_measurements_standard_deviation = measurements.universal.constants.STANDARD_DEVIATION_MIN_MEASUREMENTS
    min_measurements_correlation = measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS
    min_standard_deviation = measurements.universal.constants.np.finfo(np.float).resolution
    min_abs_correlation = measurements.universal.constants.CORRELATION_MIN_ABS_VALUE
    max_abs_correlation = measurements.universal.constants.CORRELATION_MAX_ABS_VALUE

    # general sample data

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name'))
    def _sample_data(self):
        return measurements.universal.sample_data.SampleData(self.points, self.values, self.sample_lsm)

    def _min_measurements(self, min_measurements, default_min_measurements):
        if min_measurements is None:
            min_measurements = default_min_measurements
        min_measurements_int = int(min_measurements)
        assert min_measurements_int == min_measurements
        return min_measurements_int

    @property
    def _min_value(self):
        return self._sample_data.min_value

    # mean

    @property
    def sample_means(self):
        return self._sample_data.sample_concentration_means(min_measurements=self.min_measurements_mean)

    @property
    @overrides.overrides
    def means(self):
        data = self.sample_means
        if data.count() == len(data):
            return data.data
        else:
            raise TooFewValuesError('It was not possible to calculate all values from the sample values, because to few sample values are available.')

    # deviation

    @property
    def sample_concentration_standard_deviations(self):
        return self._sample_data.sample_concentration_standard_deviations(
            min_measurements=self.min_measurements_standard_deviation,
            min_value=0)

    @property
    def concentration_standard_deviations(self):
        data = self.sample_concentration_standard_deviations
        if data.count() == len(data):
            return data.data
        else:
            raise TooFewValuesError('It was not possible to calculate all values from the sample values, because to few sample values are available.')

    @property
    def sample_average_noise_standard_deviations(self):
        return self._sample_data.sample_average_noise_standard_deviations(
            min_measurements=self.min_measurements_standard_deviation,
            min_value=self.min_standard_deviation)

    @property
    def average_noise_standard_deviations(self):
        data = self.sample_average_noise_standard_deviations
        if data.count() == len(data):
            return data.data
        else:
            raise TooFewValuesError('It was not possible to calculate all values from the sample values, because to few sample values are available.')

    @property
    def sample_noise_standard_deviations(self):
        return self._sample_data.sample_noise_standard_deviations(
            min_measurements=self.min_measurements_standard_deviation,
            min_value=self.min_standard_deviation)

    @property
    def noise_standard_deviations(self):
        data = self.sample_noise_standard_deviations
        if data.count() < len(data):
            data[data.mask] = self.average_noise_standard_deviations[data.mask]
        assert data.count() == len(data)
        return data.data

    @property
    @overrides.overrides
    def standard_deviations(self):
        standard_deviations = (self.concentration_standard_deviations**2 + self.noise_standard_deviations**2)**(0.5)
        return standard_deviations
    # correlation

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.min_measurements_correlation',
        'self.min_abs_correlation',
        'self.max_abs_correlation',
        'self.matrix_format_correlation',
        'self.dtype_correlation'))
    def _sample_correlation_object(self):
        return measurements.universal.sample_data.SampleCorrelationMatrix(
            self, self.sample_lsm, self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            matrix_format=self.matrix_format_correlation,
            dtype=self.dtype_correlation)

    @property
    def correlations_own_sample_matrix(self):
        return self._sample_correlation_object.correlation_matrix


class MeasurementsAnnualPeriodic(MeasurementsAnnualPeriodicBase):

    POSSIBLE_FILL_STRATEGIES = ('auto', 'point_average', 'lsm_average', 'constant', 'interpolate')
    POSSIBLE_KINDS = ('concentration_means', 'concentration_quantiles', 'average_noise_quantiles', 'concentration_standard_deviations', 'average_noise_standard_deviations')

    def __init__(self, tracer=None, data_set_name=None):
        self._interpolator_options = {}
        self._constant_fill_values = {}
        super().__init__(tracer=tracer, data_set_name=data_set_name)

    # interpolater

    def _check_kind(self, kind):
        if kind not in self.POSSIBLE_KINDS:
            raise ValueError('Kind must be in {}, but it is {}.'.format(self.POSSIBLE_KINDS, kind))
        return kind

    def get_interpolator_options(self, kind):
        try:
            return self._interpolator_options[kind]
        except KeyError:
            return (1, 1, 0, 0)

    def set_interpolator_options(self, kind, value):
        self._interpolator_options[self._check_kind(kind)] = value

    @property
    def interpolator_scaling_values(self):
        try:
            return self._interpolator_scaling_values
        except AttributeError:
            return measurements.universal.interpolate.default_scaling_values(self.sample_lsm)

    @interpolator_scaling_values.setter
    def interpolator_scaling_values(self, value):
        self._interpolator_scaling_values = value

    @property
    def _interpolator(self):
        return measurements.universal.interpolate.Interpolator_Annual_Periodic(
            self.sample_lsm, scaling_values=self.interpolator_scaling_values)

    # fill strategy

    @property
    def fill_strategy(self):
        try:
            return self._fill_strategy
        except AttributeError:
            return 'auto'

    @fill_strategy.setter
    def fill_strategy(self, value):
        if value in self.POSSIBLE_FILL_STRATEGIES:
            self._fill_strategy = value
        else:
            raise ValueError('Fill strategy {} is unknown. Only fill strategies {} are supported.'.format(value, self.POSSIBLE_FILL_STRATEGIES))

    def get_constant_fill_value(self, kind):
        try:
            return self._constant_fill_values[kind]
        except KeyError:
            raise ValueError('Constant fill value is not set for kind {}.'.format(kind))

    def set_constant_fill_value(self, kind, value):
        self._constant_fill_values[self._check_kind(kind)] = value

    def _choose_fill_strategy(self, number_of_sample_values):
        number_of_lsm_values = self.sample_lsm.number_of_map_indices
        sample_data_fill_amount = number_of_sample_values / number_of_lsm_values
        if sample_data_fill_amount >= 0.001:
            fill_strategy = 'interpolate'
        elif number_of_sample_values >= 1:
            fill_strategy = 'point_average'
        else:
            fill_strategy = 'constant'

        util.logging.debug('{}: Chosen {} as fill strategy, since {:d} sample data are for {:%} of the sampe lsm available.'.format(self.__class__.__name__, fill_strategy, number_of_sample_values, sample_data_fill_amount))
        return fill_strategy

    def _fill_strategy_for_kind(self, kind):
        # choose fill method
        fill_strategy = self.fill_strategy
        if fill_strategy == 'auto':
            number_of_sample_values = len(self._data_map_indices_dict(kind))
            fill_strategy = self._choose_fill_strategy(number_of_sample_values)

        # return
        util.logging.debug('{}: Fill startegy to use is {}.'.format(self.__class__.__name__, fill_strategy))
        return fill_strategy

    def _fill_strategy_with_number_of_sample_values(self, number_of_sample_values):
        # choose fill method
        fill_strategy = self.fill_strategy
        if fill_strategy == 'auto':
            fill_strategy = self._choose_fill_strategy(number_of_sample_values)

        # check number of available sample values
        if number_of_sample_values == 0 and fill_strategy != 'constant':
            raise TooFewValuesError('No sample values are available. Fill method {} is not applicable.'.format(fill_strategy))

        # return
        util.logging.debug('{}: Fill startegy to use is {}.'.format(self.__class__.__name__, fill_strategy))
        return fill_strategy

    # data general

    def _data_map_indices_dict(self, kind, *args, **kwargs):
        if kind == 'concentration_means':
            data_map_indices_dict = self._sample_data.sample_concentration_means_map_indices_dict(*args, min_measurements=self.min_measurements_mean, **kwargs)
        elif kind == 'concentration_quantiles':
            data_map_indices_dict = self._sample_data.sample_concentration_quantiles_map_indices_dict(*args, **kwargs)
        elif kind == 'average_noise_quantiles':
            data_map_indices_dict = self._sample_data.sample_average_noise_quantiles_map_indices_dict(*args, **kwargs)
        elif kind == 'concentration_standard_deviations':
            data_map_indices_dict = self._sample_data.sample_concentration_standard_deviations_map_indices_dict(*args, min_measurements=self.min_measurements_standard_deviation, min_value=0, **kwargs)
        elif kind == 'average_noise_standard_deviations':
            data_map_indices_dict = self._sample_data.sample_average_noise_standard_deviations_map_indices_dict(*args, min_measurements=self.min_measurements_standard_deviation, min_value=self.min_standard_deviation, **kwargs)
        else:
            raise ValueError('Unknown kind {}.'.format(kind))
        return data_map_indices_dict

    # data for sample lsm

    def _data_for_sample_lsm(self, kind, *args, **kwargs):
        util.logging.debug('{}: Calculating {} data for sample lsm with args {} and kwargs {}.'.format(self.__class__.__name__, kind, args, kwargs))
        self._check_kind(kind)

        # get data
        data_map_indices_dict = self._data_map_indices_dict(kind, *args, **kwargs)
        map_indices_and_values = data_map_indices_dict.toarray()

        # choose fill strategy
        fill_strategy = self._fill_strategy_with_number_of_sample_values(len(map_indices_and_values))
        util.logging.debug('{}: Filling sample lsm values with fill strategy {}.'.format(self.__class__.__name__, fill_strategy))

        # apply fill_strategy
        if fill_strategy in ('point_average', 'lsm_average', 'constant'):
            if fill_strategy == 'point_average':
                sample_values = self._sample_data._convert_map_indices_dict_to_array_for_points(data_map_indices_dict, is_discard_year=True)
                fill_value = sample_values.mean()
            elif fill_strategy == 'lsm_average':
                fill_value = map_indices_and_values[:, -1].mean()
            elif fill_strategy == 'constant':
                fill_value = self.get_constant_fill_value(kind)
            lsm_values = self.sample_lsm.insert_index_values_in_map(map_indices_and_values, no_data_value=fill_value, skip_values_on_land=False)
        elif fill_strategy == 'interpolate':
            interpolator_options = self.get_interpolator_options(kind)
            lsm_values = self._interpolator.interpolate_data_for_sample_lsm_with_map_indices(map_indices_and_values, interpolator_options)
            data_mask = np.logical_not(np.isnan(lsm_values))
            lsm_values[data_mask] = np.maximum(lsm_values[data_mask], self._min_value)
        else:
            raise ValueError('Unknown fill method {}.'.format(fill_strategy))

        assert lsm_values.shape == self.sample_lsm.dim
        return lsm_values

    def means_for_sample_lsm(self):
        values = self._data_for_sample_lsm('concentration_means')
        assert values.shape == self.sample_lsm.dim
        return values

    def concentration_quantiles_for_sample_lsm(self, quantile, min_measurements=None):
        min_measurements = self._min_measurements(min_measurements, self.min_measurements_quantile)
        values = self._data_for_sample_lsm('concentration_quantiles', quantile, min_measurements=min_measurements)
        assert values.shape == self.sample_lsm.dim
        return values

    def average_noise_quantiles_for_sample_lsm(self, quantile, min_measurements=None):
        min_measurements = self._min_measurements(min_measurements, self.min_measurements_quantile)
        values = self._data_for_sample_lsm('average_noise_quantiles', quantile, min_measurements=min_measurements)
        assert values.shape == self.sample_lsm.dim
        return values

    def concentration_standard_deviations_for_sample_lsm(self):
        values = self._data_for_sample_lsm('concentration_standard_deviations')
        assert values.shape == self.sample_lsm.dim
        return values

    def average_noise_standard_deviations_for_sample_lsm(self):
        values = self._data_for_sample_lsm('average_noise_standard_deviations')
        assert values.shape == self.sample_lsm.dim
        return values

    def standard_deviations_for_sample_lsm(self):
        values = (self.concentration_standard_deviations_for_sample_lsm()**2 + self.average_noise_standard_deviations_for_sample_lsm()**2)**(0.5)
        assert values.shape == self.sample_lsm.dim
        return values

    def concentration_relative_standard_deviations_for_sample_lsm(self):
        standard_deviations = self.concentration_standard_deviations_for_sample_lsm()
        means = self.means_for_sample_lsm()
        with np.errstate(divide='ignore'):
            values = standard_deviations / np.abs(means)
        assert values.shape == self.sample_lsm.dim
        return values

    def relative_standard_deviations_for_sample_lsm(self):
        standard_deviations = self.standard_deviations_for_sample_lsm()
        means = self.means_for_sample_lsm()
        with np.errstate(divide='ignore'):
            values = standard_deviations / np.abs(means)
        assert values.shape == self.sample_lsm.dim
        return values

    def concentration_interquartile_range_for_sample_lsm(self, min_measurements=None):
        q_25 = self.concentration_quantiles_for_sample_lsm(0.25, min_measurements=min_measurements)
        q_75 = self.concentration_quantiles_for_sample_lsm(0.75, min_measurements=min_measurements)
        values = q_75 - q_25
        assert values.shape == self.sample_lsm.dim
        return values

    def average_noise_interquartile_range_for_sample_lsm(self, min_measurements=None):
        q_25 = self.average_noise_quantiles_for_sample_lsm(0.25, min_measurements=min_measurements)
        q_75 = self.average_noise_quantiles_for_sample_lsm(0.75, min_measurements=min_measurements)
        values = q_75 - q_25
        assert values.shape == self.sample_lsm.dim
        return values

    def concentration_quartile_coefficient_of_dispersion_for_sample_lsm(self, min_measurements=None):
        # get min measurements
        try:
            len(min_measurements)
        except TypeError:
            min_measurements_q_25 = min_measurements
            min_measurements_q_50 = min_measurements
            min_measurements_q_75 = min_measurements
        else:
            assert len(min_measurements) == 3
            (min_measurements_q_25, min_measurements_q_50, min_measurements_q_75) = min_measurements
        # calculate
        q_25 = self.concentration_quantiles_for_sample_lsm(0.25, min_measurements=min_measurements_q_25)
        q_50 = self.concentration_quantiles_for_sample_lsm(0.50, min_measurements=min_measurements_q_50)
        q_75 = self.concentration_quantiles_for_sample_lsm(0.75, min_measurements=min_measurements_q_75)
        with np.errstate(divide='ignore'):
            values = (q_75 - q_25) / q_50
        assert values.shape == self.sample_lsm.dim
        return values

    # data for sample points

    def _data_for_sample_points(self, kind, *args, **kwargs):
        util.logging.debug('{}: Calculating {} data for sample points with args {} and kwargs {}.'.format(self.__class__.__name__, kind, args, kwargs))
        self._check_kind(kind)

        # get data
        data_map_indices_dict = self._data_map_indices_dict(kind, *args, **kwargs)
        data = self._sample_data._convert_map_indices_dict_to_array_for_points(data_map_indices_dict, is_discard_year=True)
        number_of_values = data.count()
        number_of_points = len(data)

        util.logging.debug('{}: Got values for {:d} of {:d} points with sample data.'.format(self.__class__.__name__, number_of_values, number_of_points))

        # fill if empty values
        if number_of_values < number_of_points:
            # choose fill strategy
            fill_strategy = self._fill_strategy_with_number_of_sample_values(len(data_map_indices_dict))
            util.logging.debug('{}: Filling remaining {:%} sample points values with fill strategy {}.'.format(self.__class__.__name__, 1 - number_of_values / number_of_points, fill_strategy))

            # fill
            if fill_strategy == 'point_average':
                data[data.mask] = data.mean()
            elif fill_strategy == 'lsm_average':
                data[data.mask] = data_map_indices_dict.values().mean()
            elif fill_strategy == 'interpolate':
                if kind == 'concentration_means':
                    data_for_sample_lsm = self.means_for_sample_lsm(*args, **kwargs)
                elif kind == 'concentration_quantiles':
                    data_for_sample_lsm = self.concentration_quantiles_for_sample_lsm(*args, **kwargs)
                elif kind == 'concentration_standard_deviations':
                    data_for_sample_lsm = self.concentration_standard_deviations_for_sample_lsm(*args, **kwargs)
                elif kind == 'average_noise_standard_deviations':
                    data_for_sample_lsm = self.average_noise_standard_deviations_for_sample_lsm(*args, **kwargs)
                data[data.mask] = self._interpolator.interpolate_data_for_points_from_interpolated_lsm_data(data_for_sample_lsm, self.points[data.mask])
                data[data.mask] = np.maximum(data[data.mask], self._min_value)
            elif fill_strategy == 'constant':
                data[data.mask] = self.get_constant_fill_value(kind)
            else:
                raise ValueError('Unknown fill method {}.'.format(fill_strategy))

        assert data.count() == len(data)
        return data.data

    @property
    @overrides.overrides
    def means(self):
        return self._data_for_sample_points('concentration_means')

    @property
    @overrides.overrides
    def concentration_standard_deviations(self):
        return self._data_for_sample_points('concentration_standard_deviations')

    @property
    @overrides.overrides
    def average_noise_standard_deviations(self):
        return self._data_for_sample_points('average_noise_standard_deviations')


class MeasurementsNearWater(Measurements):

    _PROPERTIES_OF_BASE_MEASUREMENTS = (
        'correlation_decomposition_min_value_D',
        'correlation_decomposition_min_abs_value_L',
        'permutation_method_decomposition_correlation',
        'decomposition_type_correlations',
        'matrix_format_correlation',
        'dtype_correlation')

    def __init__(self, base_measurements, use_correlations_of_base_measurements=True):
        self.base_measurements = base_measurements
        if use_correlations_of_base_measurements is None:
            use_correlations_of_base_measurements = True
        self.use_correlations_of_base_measurements = use_correlations_of_base_measurements

    # properties
    @property
    def tracer(self):
        return self.base_measurements.tracer

    @property
    def data_set_name(self):
        if self.is_restricted:
            return measurements.universal.constants.NEAR_WATER_DATA_SET_NAME.format(
                base_data_set_name=self.base_measurements.data_set_name,
                water_lsm=self.water_lsm,
                max_box_distance_to_water=self.max_box_distance_to_water)
        else:
            return self.base_measurements.data_set_name

    @property
    def max_box_distance_to_water(self):
        try:
            return self._max_box_distance_to_water
        except AttributeError:
            return None

    @max_box_distance_to_water.setter
    def max_box_distance_to_water(self, max_box_distance_to_water):
        if max_box_distance_to_water == float('inf'):
            max_box_distance_to_water = None

        # set value of passed
        if max_box_distance_to_water is not None:
            try:
                max_box_distance_to_water = int(max_box_distance_to_water)
            except TypeError:
                raise ValueError('max_box_distance_to_water must be a non-negative integer or inf or None but it is {}.'.format(max_box_distance_to_water))
            if max_box_distance_to_water < 0:
                raise ValueError('max_box_distance_to_water must be a non-negative integer but it is {}.'.format(max_box_distance_to_water))

            self._max_box_distance_to_water = max_box_distance_to_water

        # otherwise delete value
        else:
            try:
                del self._max_box_distance_to_water
            except AttributeError:
                pass

    @max_box_distance_to_water.deleter
    def max_box_distance_to_water(self):
        del self._max_box_distance_to_water

    @property
    def is_restricted(self):
        return self.max_box_distance_to_water is not None and self.max_box_distance_to_water != np.inf

    @property
    def water_lsm(self):
        try:
            return self._water_lsm
        except AttributeError:
            self.water_lsm = self.base_measurements.sample_lsm
            return self.water_lsm

    @water_lsm.setter
    def water_lsm(self, water_lsm):
        # set value of passed
        if water_lsm is not None:
            water_lsm = water_lsm.copy()
            water_lsm.t_dim = 0
            self._water_lsm = water_lsm

        # otherwise delete value
        else:
            try:
                del self.water_lsm
            except AttributeError:
                pass

    @water_lsm.deleter
    def water_lsm(self):
        del self._water_lsm

    # projection methods

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.max_box_distance_to_water',
        'self.water_lsm.name'))
    def near_water_projection_mask(self):
        mask = self.water_lsm.coordinates_near_water_mask(
            self.base_measurements.points,
            max_box_distance_to_water=self.max_box_distance_to_water)
        return mask

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.max_box_distance_to_water',
        'self.water_lsm.name'))
    def near_water_projection_matrix(self):
        mask = self.near_water_projection_mask

        n = mask.sum()
        m = len(mask)
        assert n <= m
        near_water_matrix = scipy.sparse.dok_matrix((n, m), dtype=np.int16)

        i = 0
        for j in range(m):
            if mask[j]:
                near_water_matrix[i, j] = 1
                i = i + 1
        assert i == n

        return near_water_matrix.tocsc()

    def _project_left_side(self, value):
        if self.is_restricted:
            try:
                value = value[self.near_water_projection_mask]
            except TypeError:
                value = self.near_water_projection_matrix @ value
        return value

    def _project_both_sides(self, value):
        if self.is_restricted:
            value = self.near_water_projection_matrix @ value @ self.near_water_projection_matrix.T
        return value

    # other methods

    @property
    @overrides.overrides
    def points(self):
        return self._project_left_side(self.base_measurements.points)

    @property
    @overrides.overrides
    def values(self):
        return self._project_left_side(self.base_measurements.values)

    @property
    @overrides.overrides
    def means(self):
        return self._project_left_side(self.base_measurements.means)

    @property
    @overrides.overrides
    def standard_deviations(self):
        return self._project_left_side(self.base_measurements.standard_deviations)

    @property
    @overrides.overrides
    def correlations_own_sample_matrix(self):
        if self.use_correlations_of_base_measurements:
            return self._project_both_sides(self.base_measurements.correlations_own_sample_matrix)
        else:
            return super().correlations_own_sample_matrix

    @overrides.overrides
    def correlations_other(self, measurements=None):
        return self._project_left_side(self.base_measurements.correlations_other(measurements=measurements))

    # handle properties of base measurement

    def __getattribute__(self, name):
        if name in super().__getattribute__('_PROPERTIES_OF_BASE_MEASUREMENTS'):
            return getattr(self.base_measurements, name)
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in self._PROPERTIES_OF_BASE_MEASUREMENTS:
            setattr(self.base_measurements, name, value)
        else:
            super().__setattr__(name, value)


class MeasurementsAnnualPeriodicNearWater(MeasurementsNearWater, MeasurementsAnnualPeriodic):

    _PROPERTIES_OF_BASE_MEASUREMENTS = (
        MeasurementsNearWater._PROPERTIES_OF_BASE_MEASUREMENTS
        + ('sample_lsm', 'fill_strategy',
           'min_measurements_mean', 'min_measurements_quantile',
           'min_measurements_standard_deviation', 'min_measurements_correlation',
           'min_standard_deviation',
           'min_abs_correlation', 'max_abs_correlation',
           'means_for_sample_lsm',
           'concentration_quantiles_for_sample_lsm', 'average_noise_quantiles_for_sample_lsm',
           'concentration_standard_deviations_for_sample_lsm', 'average_noise_standard_deviations_for_sample_lsm',
           'standard_deviations_for_sample_lsm',
           'concentration_relative_standard_deviations_for_sample_lsm', 'relative_standard_deviations_for_sample_lsm',
           'concentration_interquartile_range_for_sample_lsm', 'average_noise_interquartile_range_for_sample_lsm',
           'concentration_quartile_coefficient_of_dispersion_for_sample_lsm'))

    def __init__(self, base_measurements):
        MeasurementsNearWater.__init__(self, base_measurements)
        MeasurementsAnnualPeriodic.__init__(self)

    # data for points

    @property
    @overrides.overrides
    def concentration_standard_deviations(self):
        return self._project_left_side(self.base_measurements.concentration_standard_deviations)

    @property
    @overrides.overrides
    def noise_standard_deviations(self):
        return self._project_left_side(self.base_measurements.noise_standard_deviations)

    @property
    @overrides.overrides
    def average_noise_standard_deviations(self):
        return self._project_left_side(self.base_measurements.average_noise_standard_deviations)


class MeasurementsAnnualPeriodicUnion(MeasurementsAnnualPeriodic):

    def __init__(self, *measurements_list):
        util.logging.debug('Initiating {} with measurements {}.'.format(self.__class__.__name__, measurements_list))

        if len(measurements_list) == 0:
            raise ValueError('{} must be initiated with at least one measurement object.'.format(self.__class__.__name__))

        # sort
        measurements_list = sorted(measurements_list, key=lambda measurement: measurement.data_set_name)

        # check same tracer and data set name
        n = len(measurements_list)
        for i in range(n - 1):
            if measurements_list[i].tracer != measurements_list[i + 1].tracer:
                raise ValueError('Measurements objects with different tracers ({} and {}) are not allowed!'.format(measurements_list[i].tracer, measurements_list[i + 1].tracer))
            if measurements_list[i].data_set_name == measurements_list[i + 1].data_set_name:
                raise ValueError('Measurements objects with same tracer ({}) and same data set name ({}) are not allowed!'.format(measurements_list[i].tracer, measurements_list[i].data_set_name))

        # store
        self.measurements_list = measurements_list

        # chose values for union
        tracer = measurements_list[0].tracer
        data_set_name = ','.join(map(lambda measurement: measurement.data_set_name, measurements_list))
        super().__init__(tracer=tracer, data_set_name=data_set_name)

    @property
    @overrides.overrides
    def points(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.points, self.measurements_list)), axis=0)

    @property
    @overrides.overrides
    def values(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.values, self.measurements_list)))

    @property
    @overrides.overrides
    def number_of_measurements(self):
        return sum(map(lambda measurement: measurement.number_of_measurements, self.measurements_list))


class MeasurementsCollection(Measurements):

    def __init__(self, *measurements_list):
        util.logging.debug('Initiating {} with measurements {}.'.format(self.__class__.__name__, measurements_list))

        if len(measurements_list) == 0:
            raise ValueError('There are no measurements in the measurements list!')

        # remove empty measurements
        measurements_list = (measurements for measurements in measurements_list if measurements.number_of_measurements > 0)

        # sort
        measurements_list = sorted(measurements_list, key=lambda measurement: measurement.data_set_name)
        measurements_list = sorted(measurements_list, key=lambda measurement: measurement.tracer)

        # check same tracer and data set name
        n = len(measurements_list)
        for i in range(n - 1):
            if measurements_list[i].tracer == measurements_list[i + 1].tracer and measurements_list[i].data_set_name == measurements_list[i + 1].data_set_name:
                raise ValueError('There is more then one measurements object with tracer {} and data set name {}!'.format(measurements_list[i].tracer, measurements_list[i].data_set_name))

        # store
        self._measurements_list = measurements_list

    @property
    def measurements_list(self):
        return self._measurements_list

    def __str__(self):
        return ','.join(map(str, self.measurements_list))

    def __iter__(self):
        return self.measurements_list.__iter__()

    @property
    def tracer(self):
        return tuple(measurements_object.tracer for measurements_object in self)

    @property
    def tracer_str(self):
        tracer_str = ','.join(map(lambda measurement: measurement.tracer, self.measurements_list))
        return tracer_str

    @property
    def data_set_name(self):
        return tuple(measurements_object.data_set_name for measurements_object in self)

    @property
    def data_set_name_str(self):
        measurements_list = self.measurements_list
        data_set_name_str = ','.join(map(lambda measurement: measurement.data_set_name, measurements_list))
        if len(measurements_list) > 1:
            data_set_name_str = ','.join(map(lambda measurement: '{tracer}:{data_set_name}'.format(tracer=measurement.tracer, data_set_name=measurement.data_set_name), measurements_list))
        else:
            data_set_name_str = measurements_list[0].data_set_name
        return data_set_name_str

    @property
    @overrides.overrides
    def points(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.points, self.measurements_list)), axis=0)

    @property
    @overrides.overrides
    def values(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.values, self.measurements_list)))

    @property
    @overrides.overrides
    def number_of_measurements(self):
        return sum(map(lambda measurement: measurement.number_of_measurements, self.measurements_list))

    @property
    @overrides.overrides
    def means(self):
        values = np.concatenate(tuple(map(lambda measurement: measurement.means, self.measurements_list)))
        assert len(values) == self.number_of_measurements
        return values

    @property
    @overrides.overrides
    def standard_deviations(self):
        values = np.concatenate(tuple(map(lambda measurement: measurement.standard_deviations, self.measurements_list)))
        assert len(values) == self.number_of_measurements
        return values

    @property
    @overrides.overrides
    def correlations_own_sample_matrix(self):
        n = len(self.measurements_list)
        correlations = np.empty([n, n], dtype=object)

        for i in range(n):
            measurements_i = self.measurements_list[i]
            correlations[i, i] = measurements_i.correlations()
            for j in range(i + 1, n):
                measurements_j = self.measurements_list[j]
                correlations[i, j] = measurements_i.correlations(measurements_j)
                correlations[j, i] = correlations[i, j].T

        correlations = scipy.sparse.bmat(correlations, format=self.matrix_format_correlation, dtype=self.dtype_correlation)
        assert correlations.shape == (self.number_of_measurements, self.number_of_measurements)
        return correlations

    @property
    @overrides.overrides
    def correlations_own_decomposition(self):
        n = len(self.measurements_list)
        # check if all pairwise correlations are zero
        correlations_all_zero = True
        for i in range(n):
            measurements_i = self.measurements_list[i]
            for j in range(i + 1, n):
                measurements_j = self.measurements_list[j]
                correlations_i_j = measurements_i.correlations(measurements_j)
                correlations_all_zero = correlations_all_zero and correlations_i_j.nnz == 0
        # stack correlations if correlation is block diagonal
        if correlations_all_zero:
            util.logging.debug('The different measurement sets are uncorrelated. Using Block diagonal decompositions.')
            dec = self.measurements_list[0].correlations_own_decomposition
            for measurements_i in self.measurements_list[1:]:
                dec_i = measurements_i.correlations_own_decomposition
                dec = dec.append_block_decomposition(dec_i)
        else:
            util.logging.debug('The different measurement sets are correlated. Calculating own decomposition.')
            dec = super().correlations_own_decomposition
        return dec

    @property
    def points_dict(self):
        def convert_function(x):
            return x.points

        points_dict = {}

        for measurement in self.measurements_list:
            tracer = measurement.tracer
            data_set_name = measurement.data_set_name

            try:
                points_dict[tracer]
            except KeyError:
                points_dict[tracer] = {}

            points_dict[tracer][data_set_name] = convert_function(measurement)

        return points_dict

    def convert_measurements_dict_to_array(self, measurements_dict):
        value_list = [measurements_dict[measurement.tracer][measurement.data_set_name] for measurement in self.measurements_list]
        return np.concatenate(value_list)

    def subset(self, tracers):
        measurements_list = [measurement for measurement in self.measurements_list if measurement.tracer in tracers]
        subset = type(self)(*measurements_list)
        return subset


# *** caches *** #

class MeasurementsAnnualPeriodicBaseCache(MeasurementsAnnualPeriodicBase):

    # *** ids *** #

    @property
    def mean_id(self):
        return measurements.universal.constants.MEAN_ID.format(
            sample_lsm=self.sample_lsm,
            min_measurements=self.min_measurements_mean)

    def quantile_id(self, quantile, min_measurements=None):
        min_measurements = self._min_measurements(min_measurements, self.min_measurements_quantile)
        return measurements.universal.constants.QUANTILE_ID.format(
            sample_lsm=self.sample_lsm,
            quantile=float(quantile),
            min_measurements=min_measurements)

    @property
    def standard_deviation_id(self):
        return measurements.universal.constants.STANDARD_DEVIATION_ID.format(
            sample_lsm=self.sample_lsm,
            min_measurements=self.min_measurements_standard_deviation,
            min_standard_deviation=self.min_standard_deviation)

    @property
    def standard_deviation_id_without_sample_lsm(self):
        seperator = measurements.universal.constants.SEPERATOR
        standard_deviation_id = seperator.join(self.standard_deviation_id.split(seperator)[1:])
        return standard_deviation_id

    @property
    def sample_correlation_id(self):
        return measurements.universal.constants.SAMPLE_CORRELATION_ID.format(
            sample_lsm=self.sample_lsm,
            min_measurements_correlation=self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            standard_deviation_id=self.standard_deviation_id_without_sample_lsm)

    @property
    def correlation_decomposition_id(self):
        if self.correlation_decomposition_min_abs_value_L == 0:
            return measurements.universal.constants.DECOMPOSITION_ID_WITHOUT_MIN_ABS_L.format(
                decomposition_type=self.decomposition_type_correlations,
                permutation_method_decomposition_correlation=self.permutation_method_decomposition_correlation,
                min_value_D=self.correlation_decomposition_min_value_D)
        else:
            return measurements.universal.constants.DECOMPOSITION_ID.format(
                decomposition_type=self.decomposition_type_correlations,
                permutation_method_decomposition_correlation=self.permutation_method_decomposition_correlation,
                min_value_D=self.correlation_decomposition_min_value_D,
                min_abs_value_L=self.correlation_decomposition_min_abs_value_L)

    @property
    def correlation_id(self):
        if self.correlation_decomposition_min_abs_value_L == 0:
            return measurements.universal.constants.CORRELATION_ID_WITHOUT_MIN_ABS_L.format(
                sample_lsm=self.sample_lsm,
                min_measurements_correlation=self.min_measurements_correlation,
                min_abs_correlation=self.min_abs_correlation,
                max_abs_correlation=self.max_abs_correlation,
                decomposition_id=self.correlation_decomposition_id,
                standard_deviation_id=self.standard_deviation_id_without_sample_lsm,
                decomposition_type=self.decomposition_type_correlations,
                permutation_method_decomposition_correlation=self.permutation_method_decomposition_correlation,
                min_value_D=self.correlation_decomposition_min_value_D)
        else:
            return measurements.universal.constants.CORRELATION_ID.format(
                sample_lsm=self.sample_lsm,
                min_measurements_correlation=self.min_measurements_correlation,
                min_abs_correlation=self.min_abs_correlation,
                max_abs_correlation=self.max_abs_correlation,
                decomposition_id=self.correlation_decomposition_id,
                standard_deviation_id=self.standard_deviation_id_without_sample_lsm)


class MeasurementsAnnualPeriodicCache(MeasurementsAnnualPeriodicBaseCache, MeasurementsAnnualPeriodic):

    # *** ids *** #

    def _fill_strategy_id(self, kind):
        # if standard deviation, merge fill strategy string for concentration and average_noise
        if kind == 'standard_deviations':
            concentration_fill_strategy = self._fill_strategy_id('concentration_standard_deviations')
            average_noise_fill_strategy = self._fill_strategy_id('average_noise_standard_deviations')
            # if same for both, use only once
            if concentration_fill_strategy == average_noise_fill_strategy:
                fill_strategy = concentration_fill_strategy
            # if both use interpolation, use only once but two interpolator_options
            elif concentration_fill_strategy.startswith('interpolate') and average_noise_fill_strategy.startswith('interpolate'):
                interpolator_options = '+'.join([','.join(map(str, self.get_interpolator_options(kind))) for kind in ('concentration_standard_deviations', 'average_noise_standard_deviations')])
                fill_strategy = measurements.universal.constants.INTERPOLATION_FILL_STRATEGY.format(
                    scaling_values=','.join(map(lambda s: '{:.3g}'.format(s), self.interpolator_scaling_values)),
                    interpolator_options=interpolator_options)
            # if different strategies, append both strategies
            else:
                fill_strategy = util.str.merge([concentration_fill_strategy, average_noise_fill_strategy])
        # else, get used fill strategy
        else:
            if kind == 'noise_standard_deviations':
                fill_strategy = self._fill_strategy_for_kind('average_' + kind)
            else:
                fill_strategy = self._fill_strategy_for_kind(kind)

            # if interpolation, append options for interpolations
            if fill_strategy == 'interpolate':
                fill_strategy = measurements.universal.constants.INTERPOLATION_FILL_STRATEGY.format(
                    scaling_values=','.join(map(lambda s: '{:.3g}'.format(s), self.interpolator_scaling_values)),
                    interpolator_options=','.join(map(str, self.get_interpolator_options(kind))))

        return fill_strategy

    @property
    @overrides.overrides
    def mean_id(self):
        return super().mean_id + '_-_fill_' + self._fill_strategy_id('concentration_means')

    @overrides.overrides
    def quantile_id(self, quantile, min_measurements=None):
        return super().quantile_id(quantile, min_measurements=min_measurements) + '_-_fill_' + self._fill_strategy_id('concentration_quantiles')

    @property
    @overrides.overrides
    def standard_deviation_id(self):
        return super().standard_deviation_id + '_-_fill_' + self._fill_strategy_id('standard_deviations')

    # *** points and values cache files *** #

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def points(self):
        return super().points

    def points_cache_file(self):
        return measurements.universal.constants.POINTS_FILE.format(tracer=self.tracer, data_set=self.data_set_name)

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def values(self):
        return super().values

    def values_cache_file(self):
        return measurements.universal.constants.VALUES_FILE.format(tracer=self.tracer, data_set=self.data_set_name)

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def measurements_dict(self):
        return super().measurements_dict

    def measurements_dict_cache_file(self):
        return measurements.universal.constants.MEASUREMENTS_DICT_FILE.format(tracer=self.tracer, data_set=self.data_set_name)

    # *** means *** #

    def _mean_cache_file(self, target):
        fill_strategy = self._fill_strategy_id('concentration_means')
        return measurements.universal.constants.MEAN_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            target=target,
            sample_lsm=self.sample_lsm,
            min_measurements=self.min_measurements_mean,
            fill_strategy=fill_strategy)

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_mean'))
    @util.cache.file.decorator()
    @overrides.overrides
    def means(self):
        return super().means

    def means_cache_file(self):
        return self._mean_cache_file('sample_points')

    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_mean'))
    @util.cache.file.decorator()
    @overrides.overrides
    def means_for_sample_lsm(self):
        return super().means_for_sample_lsm()

    def means_for_sample_lsm_cache_file(self):
        return self._mean_cache_file(str(self.sample_lsm))

    # *** quantiles *** #

    def _quantile_cache_file(self, quantile_type, target, quantile, min_measurements):
        fill_strategy = self._fill_strategy_id(quantile_type)
        quantile = float(quantile)
        return measurements.universal.constants.QUANTILE_FILE.format(
            quantile_type=quantile_type,
            tracer=self.tracer,
            data_set=self.data_set_name,
            target=target,
            sample_lsm=self.sample_lsm,
            min_measurements=min_measurements,
            fill_strategy=fill_strategy,
            quantile=quantile)

    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy'))
    @util.cache.file.decorator()
    @overrides.overrides
    def concentration_quantiles_for_sample_lsm(self, quantile, min_measurements=None):
        min_measurements = self._min_measurements(min_measurements, self.min_measurements_quantile)
        return super().concentration_quantiles_for_sample_lsm(quantile, min_measurements=min_measurements)

    def concentration_quantiles_for_sample_lsm_cache_file(self, quantile, min_measurements=None):
        min_measurements = self._min_measurements(min_measurements, self.min_measurements_quantile)
        return self._quantile_cache_file('concentration_quantiles', str(self.sample_lsm), quantile, min_measurements=min_measurements)

    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy'))
    @util.cache.file.decorator()
    @overrides.overrides
    def average_noise_quantiles_for_sample_lsm(self, quantile, min_measurements=None):
        min_measurements = self._min_measurements(min_measurements, self.min_measurements_quantile)
        return super().average_noise_quantiles_for_sample_lsm(quantile, min_measurements=min_measurements)

    def average_noise_quantiles_for_sample_lsm_cache_file(self, quantile, min_measurements=None):
        min_measurements = self._min_measurements(min_measurements, self.min_measurements_quantile)
        return self._quantile_cache_file('average_noise_quantiles', str(self.sample_lsm), quantile, min_measurements=min_measurements)

    # *** deviation *** #

    def _standard_deviations_cache_file(self, deviation_type, target):
        fill_strategy = self._fill_strategy_id(deviation_type)
        if deviation_type == 'concentration_standard_deviations':
            min_standard_deviation = 0
        else:
            min_standard_deviation = self.min_standard_deviation
        return measurements.universal.constants.STANDARD_DEVIATION_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            min_measurements=self.min_measurements_standard_deviation,
            min_standard_deviation=min_standard_deviation,
            deviation_type=deviation_type,
            fill_strategy=fill_strategy, target=target)

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def concentration_standard_deviations(self):
        return super().concentration_standard_deviations

    def concentration_standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('concentration_standard_deviations', 'sample_points')

    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def concentration_standard_deviations_for_sample_lsm(self):
        return super().concentration_standard_deviations_for_sample_lsm()

    def concentration_standard_deviations_for_sample_lsm_cache_file(self):
        return self._standard_deviations_cache_file('concentration_standard_deviations', str(self.sample_lsm))

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def average_noise_standard_deviations(self):
        return super().average_noise_standard_deviations

    def average_noise_standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('average_noise_standard_deviations', 'sample_points')

    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def average_noise_standard_deviations_for_sample_lsm(self):
        return super().average_noise_standard_deviations_for_sample_lsm()

    def average_noise_standard_deviations_for_sample_lsm_cache_file(self):
        return self._standard_deviations_cache_file('average_noise_standard_deviations', str(self.sample_lsm))

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def noise_standard_deviations(self):
        return super().noise_standard_deviations

    def noise_standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('noise_standard_deviations', 'sample_points')

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def standard_deviations(self):
        return super().standard_deviations

    def standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('standard_deviations', 'sample_points')

    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def standard_deviations_for_sample_lsm(self):
        return super().standard_deviations_for_sample_lsm()

    def standard_deviations_for_sample_lsm_cache_file(self):
        return self._standard_deviations_cache_file('standard_deviations', str(self.sample_lsm))

    # *** correlation *** #

    @property
    @overrides.overrides
    def _sample_correlation_object(self):
        return measurements.universal.sample_data.SampleCorrelationMatrixCache(
            self, self.sample_lsm, self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            matrix_format=self.matrix_format_correlation,
            dtype=self.dtype_correlation)

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation',
        'self.min_measurements_correlation',
        'self.min_abs_correlation',
        'self.max_abs_correlation',
        'self.matrix_format_correlation',
        'self.dtype_correlation',
        'self.decomposition_type_correlations',
        'self.permutation_method_decomposition_correlation',
        'self.correlation_decomposition_min_value_D',
        'self.correlation_decomposition_min_abs_value_L'))
    @util.cache.file.decorator(load_function=matrix.decompositions.load, save_function=matrix.decompositions.save)
    @overrides.overrides
    def correlations_own_decomposition(self):
        correlation_matrix_decomposition = super().correlations_own_decomposition
        util.cache.file.save(self.correlations_own_decomposition_delta_cache_file(), correlation_matrix_decomposition.delta, np.save)
        util.cache.file.save(self.correlations_own_decomposition_omega_cache_file(), correlation_matrix_decomposition.omega, np.save)
        return correlation_matrix_decomposition

    @property
    @util.cache.file.decorator(load_function=np.load, save_function=np.save)
    def correlations_own_decomposition_delta(self):
        return self.correlations_own_decomposition.delta

    @property
    @util.cache.file.decorator(load_function=np.load, save_function=np.save)
    def correlations_own_decomposition_omega(self):
        return self.correlations_own_decomposition.omega

    def correlations_own_decomposition_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_DECOMPOSITION_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            min_measurements_correlation=self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            decomposition_id=self.correlation_decomposition_id,
            standard_deviation_id=self.standard_deviation_id_without_sample_lsm,
            dtype=self.dtype_correlation)

    def correlations_own_decomposition_delta_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_DECOMPOSITION_DELTA_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            min_measurements_correlation=self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            decomposition_id=self.correlation_decomposition_id,
            standard_deviation_id=self.standard_deviation_id_without_sample_lsm)

    def correlations_own_decomposition_omega_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_DECOMPOSITION_OMEGA_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            min_measurements_correlation=self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            decomposition_id=self.correlation_decomposition_id,
            standard_deviation_id=self.standard_deviation_id_without_sample_lsm)

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation',
        'self.min_measurements_correlation',
        'self.min_abs_correlation',
        'self.max_abs_correlation',
        'self.matrix_format_correlation',
        'self.dtype_correlation',
        'self.decomposition_type_correlations',
        'self.permutation_method_decomposition_correlation',
        'self.correlation_decomposition_min_value_D',
        'self.correlation_decomposition_min_abs_value_L'))
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own(self):
        return super().correlations_own

    def correlations_own_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_POSITIVE_DEFINITE_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            min_measurements_correlation=self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            decomposition_id=self.correlation_decomposition_id,
            standard_deviation_id=self.standard_deviation_id_without_sample_lsm,
            dtype=self.dtype_correlation,
            matrix_format=self.matrix_format_correlation)

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.sample_lsm.name',
        'self.fill_strategy',
        'self.min_measurements_standard_deviation',
        'self.min_standard_deviation',
        'self.min_measurements_correlation',
        'self.min_abs_correlation',
        'self.max_abs_correlation',
        'self.dtype_correlation',
        'self.permutation_method_decomposition_correlation'))
    @util.cache.file.decorator(load_function=np.load, save_function=np.save)
    def correlations_own_permutation_vector(self):
        return super().correlations_own_permutation_vector

    def correlations_own_permutation_vector_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_PERMUTATION_VECTOR_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            min_measurements_correlation=self.min_measurements_correlation,
            min_abs_correlation=self.min_abs_correlation,
            max_abs_correlation=self.max_abs_correlation,
            permutation_method_decomposition_correlation=self.permutation_method_decomposition_correlation,
            standard_deviation_id=self.standard_deviation_id_without_sample_lsm,
            dtype=self.dtype_correlation)


class MeasurementsAnnualPeriodicNearWaterCache(MeasurementsAnnualPeriodicCache, MeasurementsAnnualPeriodicNearWater):

    # *** ids *** #

    @property
    @overrides.overrides
    def mean_id(self):
        return self.base_measurements.mean_id

    @overrides.overrides
    def quantile_id(self, quantile, min_measurements=None):
        return self.base_measurements.quantile_id(quantile, min_measurements=min_measurements)

    @property
    @overrides.overrides
    def standard_deviation_id(self):
        return self.base_measurements.standard_deviation_id

    @property
    @overrides.overrides
    def correlation_id(self):
        return self.base_measurements.correlation_id

    # *** cacheable properties *** #

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.max_box_distance_to_water',
        'self.water_lsm.name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def near_water_projection_mask(self):
        return super().near_water_projection_mask

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.max_box_distance_to_water',
        'self.water_lsm.name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def near_water_projection_matrix(self):
        return super().near_water_projection_matrix

    @property
    @util.cache.memory.method_decorator(dependency=(
        'self.tracer',
        'self.data_set_name',
        'self.max_box_distance_to_water',
        'self.water_lsm.name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own_sample_matrix(self):
        return super().correlations_own_sample_matrix

    # *** cache files *** #

    def near_water_projection_mask_cache_file(self):
        return measurements.universal.constants.NEAR_WATER_PROJECTION_MASK_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            water_lsm=self.water_lsm,
            max_box_distance_to_water=self.max_box_distance_to_water)

    def near_water_projection_matrix_cache_file(self):
        return measurements.universal.constants.NEAR_WATER_PROJECTION_MATRIX_FILE.format(
            tracer=self.tracer,
            data_set=self.data_set_name,
            sample_lsm=self.sample_lsm,
            water_lsm=self.water_lsm,
            max_box_distance_to_water=self.max_box_distance_to_water,
            matrix_format='csc')

    @overrides.overrides
    def points_cache_file(self):
        return self.base_measurements.points_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def values_cache_file(self):
        return self.base_measurements.values_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def measurements_dict_cache_file(self):
        return self.base_measurements.measurements_dict_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def means_cache_file(self):
        return self.base_measurements.means_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def concentration_standard_deviations_cache_file(self):
        return self.base_measurements.concentration_standard_deviations_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def noise_standard_deviations_cache_file(self):
        return self.base_measurements.noise_standard_deviations_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def average_noise_standard_deviations_cache_file(self):
        return self.base_measurements.average_noise_standard_deviations_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def standard_deviations_cache_file(self):
        return self.base_measurements.standard_deviations_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    def correlations_own_sample_matrix_cache_file(self):
        return self.base_measurements._sample_correlation_object.correlation_matrix_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def correlations_own_decomposition_cache_file(self):
        return self.base_measurements.correlations_own_decomposition_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def correlations_own_decomposition_delta_cache_file(self):
        return self.base_measurements.correlations_own_decomposition_delta_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def correlations_own_decomposition_omega_cache_file(self):
        return self.base_measurements.correlations_own_decomposition_omega_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def correlations_own_cache_file(self):
        return self.base_measurements.correlations_own_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)


class MeasurementsAnnualPeriodicUnionCache(MeasurementsAnnualPeriodicUnion, MeasurementsAnnualPeriodicCache):
    pass


class MeasurementsCollectionCache(MeasurementsCollection):

    # ids

    @property
    def mean_id(self):
        return util.str.merge([measurement.mean_id for measurement in self.measurements_list])

    def quantile_id(self, quantile, min_measurements=None):
        return util.str.merge([measurement.quantile_id(quantile, min_measurements=min_measurements) for measurement in self.measurements_list])

    @property
    def standard_deviation_id(self):
        return util.str.merge([measurement.standard_deviation_id for measurement in self.measurements_list])

    @property
    def correlation_id(self):
        return util.str.merge([measurement.correlation_id for measurement in self.measurements_list])

    # *** cached values *** #

    @property
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own_sample_matrix(self):
        return super().correlations_own_sample_matrix

    @property
    @util.cache.file.decorator(load_function=matrix.decompositions.load, save_function=matrix.decompositions.save)
    @overrides.overrides
    def correlations_own_decomposition(self):
        return super().correlations_own_decomposition

    @property
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own(self):
        return super().correlations_own

    # *** files *** #

    def _merge_files(self, directory, files):
        # common dirnames above file
        number_of_measurement_dirs_below_base_dir = measurements.universal.constants.MEASUREMENT_DIR[len(measurements.universal.constants.BASE_DIR):].count(os.path.sep)
        filenames = [file[len(measurements.universal.constants.BASE_DIR):] for file in files]
        filenames = [os.path.join(*file.split(os.path.sep)[number_of_measurement_dirs_below_base_dir + 1:]) for file in filenames]

        # join dirs and filename
        filename_joined = util.str.merge(filenames)
        file_joined = os.path.join(directory, filename_joined)
        return file_joined

    @property
    def measurements_dir(self):
        return measurements.universal.constants.MEASUREMENT_DIR.format(tracer=self.tracer_str, data_set=self.data_set_name_str)

    def correlations_own_sample_matrix_cache_file(self):
        return self._merge_files(self.measurements_dir, [measurement._sample_correlation_object.correlation_matrix_cache_file() for measurement in self.measurements_list])

    def correlations_own_decomposition_cache_file(self):
        return self._merge_files(self.measurements_dir, [measurement.correlations_own_decomposition_cache_file() for measurement in self.measurements_list])

    def correlations_own_decomposition_delta_cache_file(self):
        return self._merge_files(self.measurements_dir, [measurement.correlations_own_decomposition_delta_cache_file() for measurement in self.measurements_list])

    def correlations_own_decomposition_omega_cache_file(self):
        return self._merge_files(self.measurements_dir, [measurement.correlations_own_decomposition_omega_cache_file() for measurement in self.measurements_list])

    def correlations_own_cache_file(self):
        return self._merge_files(self.measurements_dir, [measurement.correlations_own_cache_file() for measurement in self.measurements_list])


# *** generic *** #

class TooFewValuesError(Exception):

    def __init__(self, message=None):
        if message is None:
            message = 'Too few values are available.'
        super().__init__(message)


def as_measurements_collection(measurements_object):
    if isinstance(measurements_object, MeasurementsCollectionCache):
        return measurements_object
    else:
        try:
            len(measurements_object)
        except TypeError:
            return MeasurementsCollectionCache(measurements_object)
        else:
            return MeasurementsCollectionCache(*measurements_object)
