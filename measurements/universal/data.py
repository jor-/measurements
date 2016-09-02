import abc
import collections

import numpy as np
import scipy.sparse

import util.cache.file_based
import util.cache.memory_based
import util.options
import util.logging

import measurements.universal.dict
import measurements.universal.interpolate
import measurements.universal.sample_data
import measurements.universal.constants

logger = util.logging.logger



class TooFewValuesError(Exception):
    
    def __init__(self):
        message = 'Too few values are available.'
        super().__init__(message)



class Measurements():
    
    def __init__(self, tracer=None, data_set_name=None):
        self._tracer = tracer
        self._data_set_name = data_set_name
        self.cholesky_ordering_method_correlation = 'best'
        logger.debug('{}: initialized with tracer {} and data set {}.'.format(self.__class__.__name__, tracer, data_set_name))
    
    
    def __repr__(self):
        return '<measurements for tracer "{tracer}" with data set "{data_set_name}" and {number_of_measurements} measurements>'.format(tracer=self.tracer, data_set_name=self.data_set_name, number_of_measurements=self.number_of_measurements)
    
    def __str__(self):
        return '{tracer}:{data_set_name}'.format(tracer=self.tracer, data_set_name=self.data_set_name)


    @property
    def tracer(self):
        if self._tracer is not None:
            return self._tracer
        else:
            return ValueError('Tracer is not set.')
    
    @property
    def data_set_name(self):
        if self._data_set_name is not None:
            return self._data_set_name
        else:
            return ValueError('Data set name is not set.')

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

    @property
    @util.cache.memory_based.decorator()
    def measurements_dict(self):
        m = measurements.universal.dict.MeasurementsDict()
        m.append_values(self.points, self.values)
        return m
    

    @property
    @abc.abstractmethod
    def means(self):
        raise NotImplementedError()    

    @property
    @abc.abstractmethod
    def standard_deviations(self):
        raise NotImplementedError()
    

    @property
    def variances(self):
        return self.standard_deviations


    @property
    def correlations_own(self):
        return scipy.sparse.eye(self.number_of_measurements)    

    @property
    def correlations_own_cholesky_decomposition(self):
        import util.math.sparse.decompose.with_cholmod
        P, L = util.math.sparse.decompose.with_cholmod.cholesky(self.correlations_own, ordering_method=self.cholesky_ordering_method_correlation, return_type=util.math.sparse.decompose.with_cholmod.RETURN_P_L)
        return {'P': P, 'L': L}
    
    def correlations_other(self, measurements=None):
        return scipy.sparse.dia_matrix((self.number_of_measurements, measurements.number_of_measurements))
        

    def correlations(self, measurements=None):
        if measurements is None or measurements == self:
            return self.correlations_own
        else:
            return self.correlations_other(measurements=measurements)



class MeasurementsAnnualPeriodicBase(Measurements):
    
    def __init__(self, sample_lsm, *args, min_deviation=np.finfo(np.float).resolution, min_abs_correlation=measurements.universal.constants.CORRELATION_MIN_ABS_VALUE, max_abs_correlation=measurements.universal.constants.CORRELATION_MAX_ABS_VALUE, min_measurements_means=measurements.universal.constants.MEAN_MIN_MEASUREMENTS, min_measurements_standard_deviations=measurements.universal.constants.DEVIATION_MIN_MEASUREMENTS, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS, **kargs):
        
        super().__init__(*args, **kargs)
        
        self._sample_lsm = sample_lsm
        
        self.min_deviation = min_deviation
        self.min_abs_correlation = min_abs_correlation
        self.max_abs_correlation = max_abs_correlation
        
        self.min_measurements_means = min_measurements_means
        self.min_measurements_standard_deviations = min_measurements_standard_deviations
        self.min_measurements_correlations = min_measurements_correlations
        self.min_measurements_for_fill = 1

        self.cholesky_min_diag_value_correlation = measurements.universal.constants.CORRELATION_CHOLESKY_MIN_DIAG_VALUE
        self.cholesky_ordering_method_correlation = measurements.universal.constants.CORRELATION_CHOLESKY_ORDERING_METHOD
        self.cholesky_reordering_correlation = measurements.universal.constants.CORRELATION_CHOLEKSY_REORDER_AFTER_EACH_STEP
        self.dtype_correlation = measurements.universal.constants.CORRELATION_DTYPE
        self.matrix_format_correlation = measurements.universal.constants.CORRELATION_FORMAT
    

    @property
    @util.cache.memory_based.decorator()
    def _sample_mean_and_deviation(self):
        return measurements.universal.sample_data.SampleMeanAndDeviation(self.points, self.values, self.sample_lsm)

    @property
    @util.cache.memory_based.decorator()
    def _sample_correlation(self):
        return measurements.universal.sample_data.SampleCorrelationMatrix(self, self.sample_lsm, self.min_measurements_correlations, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, matrix_format=self.matrix_format_correlation, dtype=self.dtype_correlation)

    @property
    def sample_lsm(self):
        return self._sample_lsm

    
    def _fill(self, data, fill_function):
        assert callable(fill_function)
        
        number_of_values = data.count()
        number_of_points = len(data)

        logger.debug('{}: Got values for {:d} of {:d} points with sample data. Filling remaining {:%}.'.format(self.__class__.__name__, number_of_values, number_of_points, 1-number_of_values/number_of_points))
        if number_of_values < number_of_points:
            if number_of_values >= self.min_measurements_for_fill:
                data = fill_function(data)
            else:
                raise TooFewValuesError()
        
        assert data.count() == len(data)
        return data.data
    

    ## mean

    @abc.abstractmethod
    def _fill_means(self, data):
        raise NotImplementedError()
    
    @property
    @util.cache.memory_based.decorator()
    def sample_means(self):
        return self._sample_mean_and_deviation.sample_concentration_means(min_measurements=self.min_measurements_means)

    @property
    @util.cache.memory_based.decorator()
    def means(self):
        means = self._fill(self.sample_means, self._fill_means)
        return means
    

    ## deviation
    
    @abc.abstractmethod
    def _fill_concentration_standard_deviations(self, data):
        raise NotImplementedError()
    
    @property
    @util.cache.memory_based.decorator()
    def sample_concentration_standard_deviations(self):
        return self._sample_mean_and_deviation.sample_concentration_standard_deviations(min_measurements=self.min_measurements_standard_deviations, min_deviation=0)

    @property
    @util.cache.memory_based.decorator()
    def concentration_standard_deviations(self):
        return self._fill(self.sample_concentration_standard_deviations, self._fill_concentration_standard_deviations)

    @abc.abstractmethod
    def _fill_noise_standard_deviations(self, data):
        raise NotImplementedError()
    
    @property
    @util.cache.memory_based.decorator()
    def sample_noise_standard_deviations(self):
        return self._sample_mean_and_deviation.sample_noise_standard_deviations(min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation)

    @property
    @util.cache.memory_based.decorator()
    def noise_standard_deviations(self):
        return self._fill(self.sample_noise_standard_deviations, self._fill_noise_standard_deviations)
    
    @property
    def sample_average_noise_standard_deviations(self):
        return self._sample_mean_and_deviation.sample_average_noise_standard_deviations(min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation)

    @property
    def standard_deviations(self):
        standard_deviations = (self.concentration_standard_deviations**2 + self.noise_standard_deviations**2)**(1/2)
        return standard_deviations
    

    ## correlation

    @property
    def correlations_own_sample_matrix(self):
        return self._sample_correlation.correlation_matrix
    
    
    @property
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation)
        return correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)


    @property
    def correlations_own_cholesky_decomposition(self):
        import util.math.sparse.decompose.with_cholmod
        P, L = util.math.sparse.decompose.with_cholmod.cholesky(self.correlations_own, ordering_method=self.cholesky_ordering_method_correlation, return_type=util.math.sparse.decompose.with_cholmod.RETURN_P_L)
        return {'P': P.asformat(self.matrix_format_correlation).astype(np.int8), 'L': L.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)}



class MeasurementsAnnualPeriodicFillAverage(MeasurementsAnnualPeriodicBase):
    
    def _fill_means(self, data):
        data_average = data.mean()
        data[data.mask] = data_average
        return data
    

    def _fill_concentration_standard_deviations(self, data):
        data_average = data.mean()
        data[data.mask] = data_average
        return data    


    def _fill_noise_standard_deviations(self, data):
        sample_average_noise_standard_deviations = self.sample_average_noise_standard_deviations
        data[data.mask] = sample_average_noise_standard_deviations[data.mask]
        data[data.mask] = sample_average_noise_standard_deviations.mean()
        return data



class MeasurementsAnnualPeriodicFillAverageWithDeviationRatio(MeasurementsAnnualPeriodicFillAverage):
    
    def __init__(self, *args, standard_deviation_concentration_noise_ratio=None, **kargs):
        self.standard_deviation_concentration_noise_ratio = standard_deviation_concentration_noise_ratio
        super().__init__(*args, **kargs)
    
    
    @property
    @util.cache.memory_based.decorator()
    def concentration_standard_deviations(self):
        try:
            concentration_standard_deviations = super().concentration_standard_deviations
        except TooFewValuesError as e:
            logger.debug('{}: Too few values to calculate some sample standard_deviations for the concentration. Trying to use the concentration noise ratio {}.'.format(self.__class__.__name__, self.standard_deviation_concentration_noise_ratio))
            if self.standard_deviation_concentration_noise_ratio is not None:
                noise_standard_deviations = super().noise_standard_deviations
                concentration_standard_deviations = noise_standard_deviations * self.standard_deviation_concentration_noise_ratio
            else:
                raise e
        return concentration_standard_deviations
    
    @property
    @util.cache.memory_based.decorator()
    def noise_standard_deviations(self):
        try:
            noise_standard_deviations = super().noise_standard_deviations
        except TooFewValuesError as e:
            logger.debug('{}: Too few values to calculate some sample standard_deviations for the concentration. Trying to use the concentration noise ratio {}.'.format(self.__class__.__name__, self.standard_deviation_concentration_noise_ratio))
            if self.standard_deviation_concentration_noise_ratio is not None:
                concentration_standard_deviations = super().concentration_standard_deviations
                noise_standard_deviations = concentration_standard_deviations / self.standard_deviation_concentration_noise_ratio
            else:
                raise e
        return noise_standard_deviations



class MeasurementsAnnualPeriodicFillInterpolation(MeasurementsAnnualPeriodicBase):
    
    def __init__(self, mean_interpolator_setup, concentation_standard_deviation_interpolator_setup, average_noise_standard_deviation_interpolator_setup, *args, **kargs):
        
        super().__init__(*args, **kargs)
        
        self.mean_interpolator_setup = mean_interpolator_setup
        self.concentation_standard_deviation_interpolator_setup = concentation_standard_deviation_interpolator_setup
        self.average_noise_standard_deviation_interpolator_setup = average_noise_standard_deviation_interpolator_setup
        self.scaling_values = measurements.universal.interpolate.default_scaling_values(self.sample_lsm)
        self.interpolator = measurements.universal.interpolate.Interpolator_Annual_Periodic(self.sample_lsm, scaling_values=self.scaling_values)
    
    
    ## interpolate for sample lsm
    
    def _interpolate_for_sample_lsm(self, data, interpolator_setup):
        sample_points_and_values = np.concatenate((self.points[~data.mask], data[~data.mask,np.newaxis]), axis=1)
        interpolated_values = self.interpolator.interpolate_data_for_lsm(sample_points_and_values, self.sample_lsm, interpolator_setup)
        return interpolated_values
    

    @property
    @util.cache.memory_based.decorator()
    def means_for_sample_lsm(self):
        return self._interpolate_for_sample_lsm(self.sample_means, self.mean_interpolator_setup)
    
    @property
    @util.cache.memory_based.decorator()
    def concentation_standard_deviation_for_sample_lsm(self):
        return self._interpolate_for_sample_lsm(self.sample_concentration_standard_deviations, self.concentation_standard_deviation_interpolator_setup)
    
    @property
    @util.cache.memory_based.decorator()
    def average_noise_standard_deviation_for_sample_lsm(self):
        return self._interpolate_for_sample_lsm(self.sample_average_noise_standard_deviations, self.average_noise_standard_deviation_interpolator_setup)

    @property
    @util.cache.memory_based.decorator()
    def standard_deviation_for_sample_lsm(self):
        return (self.concentation_standard_deviation_for_sample_lsm**2 + self.average_noise_standard_deviation_for_sample_lsm**2)**(1/2)
    
    
    ## interpolate for points
    
    def _fill_with_interpolation(self, data, interpolated_lsm_data):
        interpolated_points = self.points[data.mask]
        interpolated_values = self.interpolator.interpolate_data_for_points_from_interpolated_lsm_data(interpolated_lsm_data, interpolated_points)
        data[data.mask] = interpolated_values
        return data
    

    def _fill_means(self, data):
        return self._fill_with_interpolation(data, self.means_for_sample_lsm)


    def _fill_concentration_standard_deviations(self, data):
        return self._fill_with_interpolation(data, self.concentation_standard_deviation_for_sample_lsm)


    def _fill_noise_standard_deviations(self, data):
        sample_average_noise_standard_deviations = self.sample_average_noise_standard_deviations
        interpolated_average_noise_standard_deviations = self._fill_with_interpolation(sample_average_noise_standard_deviations, self.average_noise_standard_deviation_for_sample_lsm)
        data[data.mask] = interpolated_average_noise_standard_deviations[data.mask]
        return data



class MeasurementsAnnualPeriodicNearWater(MeasurementsAnnualPeriodicBase):
    
    def __init__(self, base_measurements, water_lsm=None, max_box_distance_to_water=0):
        self.base_measurements = base_measurements
        self.max_box_distance_to_water = max_box_distance_to_water
        
        if water_lsm is None:
            water_lsm = self.base_measurements.sample_lsm
        water_lsm = water_lsm.copy()
        water_lsm.t_dim = 0
        self.water_lsm = water_lsm
        
        tracer = self.base_measurements.tracer
        data_set_name = measurements.universal.constants.NEAR_WATER_DATA_SET_NAME.format(base_data_set_name=self.base_measurements.data_set_name, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)
        
        super().__init__(self.base_measurements.sample_lsm, min_deviation=self.base_measurements.min_deviation, min_abs_correlation=self.base_measurements.min_abs_correlation, max_abs_correlation=self.base_measurements.max_abs_correlation, min_measurements_means=self.base_measurements.min_measurements_means, min_measurements_standard_deviations=self.base_measurements.min_measurements_standard_deviations, min_measurements_correlations=self.base_measurements.min_measurements_correlations, tracer=tracer, data_set_name=data_set_name)
    
    
    ## projection methods
    
    @property
    @util.cache.memory_based.decorator()
    def near_water_projection_matrix(self):
        mask = self.water_lsm.coordinates_near_water_mask(self.base_measurements.points, max_box_distance_to_water=self.max_box_distance_to_water)
        
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
    
    
    ## other methods
    
    @property
    def points(self):
        return self.near_water_projection_matrix * self.base_measurements.points

    @property
    def values(self):
        return self.near_water_projection_matrix * self.base_measurements.values
    
    @property
    def means(self):
        return self.near_water_projection_matrix * self.base_measurements.means

    @property
    def standard_deviations(self):
        return self.near_water_projection_matrix * self.base_measurements.standard_deviations
        
    @property
    def concentration_standard_deviations(self):
        return self.near_water_projection_matrix * self.base_measurements.concentration_standard_deviations

    @property
    def noise_standard_deviations(self):
        return self.near_water_projection_matrix * self.base_measurements.noise_standard_deviations
        
    @property
    def correlations_own_sample_matrix(self):
        return self.near_water_projection_matrix * self.base_measurements.correlations_own_sample_matrix * self.near_water_projection_matrix.T
    
    def correlations_other(self, measurements=None):
        return self.near_water_projection_matrix * self.base_measurements.correlations_other(measurements=measurements)



## cache

class MeasurementsAnnualPeriodicCacheBase(MeasurementsAnnualPeriodicBase):
    
    ## cachable properties

    @property
    def _sample_correlation(self):
        return measurements.universal.sample_data.SampleCorrelationMatrixCache(self, self.sample_lsm, self.min_measurements_correlations, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, matrix_format=self.matrix_format_correlation, dtype=self.dtype_correlation)
    

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def points(self):
        return super().points

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def values(self):
        return super().values

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def measurements_dict(self):
        return super().measurements_dict
    
    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def means(self):
        return super().means
    
    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def concentration_standard_deviations(self):
        return super().concentration_standard_deviations
    
    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def noise_standard_deviations(self):
        return super().noise_standard_deviations
    
    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def standard_deviations(self):
        return super().standard_deviations

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation, reduction_factors_file=self.reduction_factors_cache_file())
        return correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)

    
    @property
    @util.cache.memory_based.decorator()
    def _correlations_own_cholesky_decomposition(self):
        return super().correlations_own_cholesky_decomposition
    
    @property
    @util.cache.file_based.decorator()
    def _correlations_own_cholesky_decomposition_P(self):
        return self._correlations_own_cholesky_decomposition['P']
    
    @property
    @util.cache.file_based.decorator()
    def _correlations_own_cholesky_decomposition_L(self):
        return self._correlations_own_cholesky_decomposition['L']
    
    @property
    def correlations_own_cholesky_decomposition(self):
        return {'P': self._correlations_own_cholesky_decomposition_P, 'L': self._correlations_own_cholesky_decomposition_L}
    
    
    ## cache files
    
    def points_cache_file(self):
        return measurements.universal.constants.POINTS_FILE.format(tracer=self.tracer, data_set=self.data_set_name)
    
    def values_cache_file(self):
        return measurements.universal.constants.VALUES_FILE.format(tracer=self.tracer, data_set=self.data_set_name)
    
    def measurements_dict_cache_file(self):
        return measurements.universal.constants.MEASUREMENTS_DICT_FILE.format(tracer=self.tracer, data_set=self.data_set_name)
    
    @abc.abstractmethod
    def means_cache_file(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def concentration_standard_deviations_cache_file(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def noise_standard_deviations_cache_file(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def standard_deviations_cache_file(self):
        raise NotImplementedError()
    
    def reduction_factors_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_correlations, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, dtype=self.dtype_correlation, matrix_format=self.matrix_format_correlation) 
    
    def correlations_own_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_POSITIVE_DEFINITE_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_correlations, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, dtype=self.dtype_correlation, matrix_format=self.matrix_format_correlation)    
    
    def _correlations_own_cholesky_decomposition_P_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_correlations, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, dtype=np.int8, matrix_format=self.matrix_format_correlation, factor_type='P')
    
    def _correlations_own_cholesky_decomposition_L_cache_file(self):
        return measurements.universal.constants.CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_correlations, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, dtype=self.dtype_correlation, matrix_format=self.matrix_format_correlation, factor_type='L')



class MeasurementsAnnualPeriodicFillAverageCache(MeasurementsAnnualPeriodicCacheBase, MeasurementsAnnualPeriodicFillAverageWithDeviationRatio):
    
    ## cache files
    
    def means_cache_file(self):
        return measurements.universal.constants.MEAN_FILL_AVERAGED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_means)
    
    def concentration_standard_deviations_cache_file(self):
        return measurements.universal.constants.CONCENTRATION_DEVIATION_FILL_AVERAGED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation)
    
    def noise_standard_deviations_cache_file(self):
        return measurements.universal.constants.NOISE_DEVIATION_FILL_AVERAGED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation)
    
    def standard_deviations_cache_file(self):
        return measurements.universal.constants.TOTAL_DEVIATION_FILL_AVERAGED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation)
    


class MeasurementsAnnualPeriodicFillInterpolationCache(MeasurementsAnnualPeriodicCacheBase, MeasurementsAnnualPeriodicFillInterpolation):
    
    ## cacheable properties

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def means_for_sample_lsm(self):
        return super().means_for_sample_lsm
    
    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def concentation_standard_deviation_for_sample_lsm(self):
        return super().concentation_standard_deviation_for_sample_lsm
    
    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def average_noise_standard_deviation_for_sample_lsm(self):
        return super().average_noise_standard_deviation_for_sample_lsm

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def standard_deviation_for_sample_lsm(self):
        return super().standard_deviation_for_sample_lsm
    

    ## cache files
    
    def means_cache_file(self):
        return measurements.universal.constants.MEAN_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target='points', sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_means, scaling_values=self.scaling_values, interpolator_setup=self.mean_interpolator_setup).replace('(','').replace(')','').replace(' ','')
    
    def concentration_standard_deviations_cache_file(self):
        return measurements.universal.constants.CONCENTRATION_DEVIATION_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target='points', sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation, scaling_values=self.scaling_values, concentration_interpolator_setup=self.concentation_standard_deviation_interpolator_setup).replace('(','').replace(')','').replace(' ','')
    
    def noise_standard_deviations_cache_file(self):
        return measurements.universal.constants.NOISE_DEVIATION_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target='points', sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation, scaling_values=self.scaling_values, average_noise_interpolator_setup=self.average_noise_standard_deviation_interpolator_setup).replace('(','').replace(')','').replace(' ','')
    
    def standard_deviations_cache_file(self):
        return measurements.universal.constants.TOTAL_DEVIATION_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target='points', sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation, scaling_values=self.scaling_values, concentration_interpolator_setup=self.concentation_standard_deviation_interpolator_setup, average_noise_interpolator_setup=self.average_noise_standard_deviation_interpolator_setup).replace('(','').replace(')','').replace(' ','')    
    
    def means_for_sample_lsm_cache_file(self):
        return measurements.universal.constants.MEAN_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target=self.sample_lsm, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_means, scaling_values=self.scaling_values, interpolator_setup=self.mean_interpolator_setup).replace('(','').replace(')','').replace(' ','')
    
    def concentation_standard_deviation_for_sample_lsm_cache_file(self):
        return measurements.universal.constants.CONCENTRATION_DEVIATION_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target=self.sample_lsm, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation, scaling_values=self.scaling_values, concentration_interpolator_setup=self.concentation_standard_deviation_interpolator_setup).replace('(','').replace(')','').replace(' ','')
    
    def average_noise_standard_deviation_for_sample_lsm_cache_file(self):
        return measurements.universal.constants.NOISE_DEVIATION_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target=self.sample_lsm, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation, scaling_values=self.scaling_values, average_noise_interpolator_setup=self.average_noise_standard_deviation_interpolator_setup).replace('(','').replace(')','').replace(' ','')
    
    def standard_deviation_for_sample_lsm_cache_file(self):
        return measurements.universal.constants.TOTAL_DEVIATION_FILL_INTERPOLATED_FILE.format(tracer=self.tracer, data_set=self.data_set_name, interpolation_target=self.sample_lsm, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviations, min_deviation=self.min_deviation, scaling_values=self.scaling_values, concentration_interpolator_setup=self.concentation_standard_deviation_interpolator_setup, average_noise_interpolator_setup=self.average_noise_standard_deviation_interpolator_setup).replace('(','').replace(')','').replace(' ','')



class MeasurementsAnnualPeriodicNearWaterCache(MeasurementsAnnualPeriodicCacheBase, MeasurementsAnnualPeriodicNearWater):
    
    def __init__(self, base_measurements, water_lsm=None, max_box_distance_to_water=0):
        super().__init__(base_measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)
    
    
    ## cacheable properties
    
    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def near_water_projection_matrix(self):
        return super().near_water_projection_matrix
    

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def correlations_own_sample_matrix(self):
        return super().correlations_own_sample_matrix
    

    ## cache files
    
    def near_water_projection_matrix_cache_file(self):
        return measurements.universal.constants.NEAR_WATER_PROJECTION_MASK_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, water_lsm=self.water_lsm, max_box_distance_to_water=self.max_box_distance_to_water, matrix_format='csc')
    
    def points_cache_file(self):
        return self.base_measurements.points_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def values_cache_file(self):
        return self.base_measurements.values_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def measurements_dict_cache_file(self):
        return self.base_measurements.measurements_dict_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def means_cache_file(self):
        return self.base_measurements.means_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def concentration_standard_deviations_cache_file(self):
        return self.base_measurements.concentration_standard_deviations_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def noise_standard_deviations_cache_file(self):
        return self.base_measurements.noise_standard_deviations_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def standard_deviations_cache_file(self):
        return self.base_measurements.standard_deviations_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    def correlations_own_sample_matrix_cache_file(self):
        return self.base_measurements._sample_correlation.correlation_matrix_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def correlations_own_cache_file(self):
        return self.base_measurements.correlations_own_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)
    
    def correlations_own_cholesky_decomposition_cache_file(self):
        return self.base_measurements.correlations_own_cholesky_decomposition_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)



## Measurements Collection

class MeasurementsCollection(Measurements):
    
    def __init__(self, *measurements_list):
        logger.debug('Initiating {} with measurements {}.'.format(self.__class__.__name__, measurements_list))
        
        ## sort
        measurements_list = sorted(measurements_list, key=lambda measurement: measurement.data_set_name)
        measurements_list = sorted(measurements_list, key=lambda measurement: measurement.tracer)
        
        ## check same tracer and data set name
        n = len(measurements_list)
        for i in range(n-1):
            if measurements_list[i].tracer == measurements_list[i+1].tracer and measurements_list[i].data_set_name == measurements_list[i+1].data_set_name:
                raise ValueError('There is more then one measurements object with tracer {} and data set name {}!'.format(measurements_list[i].tracer, measurements_list[i].data_set_name))

        ## store
        self.measurements = measurements_list
        
        ## make tracer and data set name for collection
        tracer = ','.join(map(lambda measurement: measurement.tracer, measurements_list))
        data_set_name = ','.join(map(lambda measurement: measurement.data_set_name, measurements_list))
        super().__init__(tracer=tracer, data_set_name=data_set_name)
        
        ## correlation constants
        self.min_abs_correlation = min([measurement.min_abs_correlation for measurement in measurements_list])
        self.cholesky_min_diag_value_correlation = measurements.universal.constants.CORRELATION_CHOLESKY_MIN_DIAG_VALUE
        self.cholesky_ordering_method_correlation = measurements.universal.constants.CORRELATION_CHOLESKY_ORDERING_METHOD
        self.cholesky_reordering_correlation = measurements.universal.constants.CORRELATION_CHOLEKSY_REORDER_AFTER_EACH_STEP
        self.matrix_format_correlation = measurements.universal.constants.CORRELATION_FORMAT
        self.dtype_correlation = measurements.universal.constants.CORRELATION_DTYPE
    
    
    def __str__(self):
        return ','.join(map(str, self.measurements))
    
    
    def __iter__(self):
        return self.measurements.__iter__()
    
    
    @property
    def points(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.points, self.measurements)), axis=0)
    
    @property
    def values(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.values, self.measurements)))
    

    @property
    def number_of_measurements(self):
        return sum(map(lambda measurement: measurement.number_of_measurements, self.measurements))
    
    @property
    def means(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.means, self.measurements)))  

    @property
    def standard_deviations(self):
        return np.concatenate(tuple(map(lambda measurement: measurement.standard_deviations, self.measurements)))
    
    
    @property
    def correlations_own_sample_matrix(self):
        n = len(self.measurements)
        correlations = np.array([n,n], dtype=object)
        
        for i in range(n):
            measurements_i = self.measurements[i]
            correlations[i, i] = measurements_i.correlations()
            for j in range(i+1, n):
                measurements_j = self.measurements[j]
                correlations[i, j] = measurements_i.correlations(measurements_j)
                correlations[j, i] = correlations[i, j].T
        
        correlations = scipy.sparse.bmat(correlations, format=self.matrix_format_correlation, dtype=self.dtype_correlation)
        return correlations

    
    @property
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation)
        return correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)


    @property
    def correlations_own_cholesky_decomposition(self):
        import util.math.sparse.decompose.with_cholmod
        P, L = util.math.sparse.decompose.with_cholmod.cholesky(self.correlations_own, ordering_method=self.cholesky_ordering_method_correlation, return_type=util.math.sparse.decompose.with_cholmod.RETURN_P_L)
        return {'P': P.asformat(self.matrix_format_correlation).astype(np.int8), 'L': L.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)}


    def _measurements_dict(self, convert_function=None):
        if convert_function is None:
            convert_function = lambda x: x
        
        results = {}
        
        for measurement in self.measurements:
            tracer = measurement.tracer
            data_set_name = measurement.data_set_name
            
            try:
                results[tracer]
            except KeyError:
                results[tracer] = {}
            
            results[tracer][data_set_name] = convert_function(measurement)
        
        return results
    
    
    @property
    def points_dict(self):
        return self._measurements_dict(convert_function=lambda m: m.points)
    
    
    def convert_measurements_dict_to_array(self, measurements_dict):
        value_list = [measurements_dict[measurement.tracer][measurement.data_set_name] for measurement in self.measurements]
        return np.concatenate(value_list)



class MeasurementsCollectionCache(MeasurementsCollection):
    
    def _merge_files(self, files):
        filenames = [os.path.basename(file) for file in files]
        
        filenames_base = [os.path.splitext(filename)[0] for filename in filenames]
        filename_base = ','.join(filenames_base)
        
        filenames_ext = [os.path.splitext(filename)[1] for filename in filenames]
        filename_ext = filenames_ext[0]
        assert filenames_ext.count(filename_ext) == len(filenames_ext)
        
        filename = filename_base + filename_ext
        return filename

    
    @property
    @util.cache.file_based.decorator()
    def correlations_own_sample_matrix(self):
        return super().correlations_own_sample_matrix

    @property
    @util.cache.file_based.decorator()
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation, reduction_factors_file=self.reduction_factors_file)
        return correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)

    @property
    @util.cache.memory_based.decorator()
    def _correlations_own_cholesky_decomposition(self):
        return super().correlations_own_cholesky_decomposition
    
    @property
    @util.cache.file_based.decorator()
    def _correlations_own_cholesky_decomposition_P(self):
        return self._correlations_own_cholesky_decomposition['P']
    
    @property
    @util.cache.file_based.decorator()
    def _correlations_own_cholesky_decomposition_L(self):
        return self._correlations_own_cholesky_decomposition['L']
    
    @property
    def correlations_own_cholesky_decomposition(self):
        return {'P': self._correlations_own_cholesky_decomposition_P, 'L': self._correlations_own_cholesky_decomposition_L}
    
    
    def reduction_factors_cache_file(self):
        return self._merge_files([measurement.reduction_factors_cache_file for measurement in self.measurements])
    
    def correlations_own_cache_file(self):
        return self._merge_files([measurement.correlations_own_cache_file for measurement in self.measurements])
    
    def _correlations_own_cholesky_decomposition_P_cache_file(self):
        return self._merge_files([measurement._correlations_own_cholesky_decomposition_P_cache_file for measurement in self.measurements])
    
    def _correlations_own_cholesky_decomposition_L_cache_file(self):
        return self._merge_files([measurement._correlations_own_cholesky_decomposition_L_cache_file for measurement in self.measurements])



def as_measurements_collection(measurements):
    if isinstance(measurements, MeasurementsCollectionCache):
        return measurements
    else:
        return MeasurementsCollection(*measurements)
    
