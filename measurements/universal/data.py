import abc
import collections

import numpy as np
import scipy.sparse
import overrides

import util.cache.file
import util.cache.memory
import util.options
import util.logging

import measurements.universal.dict
import measurements.universal.interpolate
import measurements.universal.sample_data
import measurements.universal.constants

logger = util.logging.logger


## generic

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
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
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
        return self.standard_deviations ** 2


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



## annual periodic

class MeasurementsAnnualPeriodicBase(Measurements):
    
    def __init__(self, sample_lsm, tracer=None, data_set_name=None, min_standard_deviation=np.finfo(np.float).resolution, min_abs_correlation=measurements.universal.constants.CORRELATION_MIN_ABS_VALUE, max_abs_correlation=measurements.universal.constants.CORRELATION_MAX_ABS_VALUE, min_measurements_mean=measurements.universal.constants.MEAN_MIN_MEASUREMENTS, min_measurements_standard_deviation=measurements.universal.constants.DEVIATION_MIN_MEASUREMENTS, min_measurements_correlation=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        
        super().__init__(tracer=tracer, data_set_name=data_set_name)
        
        self._sample_lsm = sample_lsm
        
        self.min_measurements_mean = min_measurements_mean
        
        self.min_measurements_standard_deviation = min_measurements_standard_deviation
        self.min_standard_deviation = min_standard_deviation
        
        self.min_measurements_correlation = min_measurements_correlation
        self.min_abs_correlation = min_abs_correlation
        self.max_abs_correlation = max_abs_correlation

        self.cholesky_min_diag_value_correlation = measurements.universal.constants.CORRELATION_CHOLESKY_MIN_DIAG_VALUE
        self.cholesky_ordering_method_correlation = measurements.universal.constants.CORRELATION_CHOLESKY_ORDERING_METHOD
        self.cholesky_reordering_correlation = measurements.universal.constants.CORRELATION_CHOLEKSY_REORDER_AFTER_EACH_STEP
        self.dtype_correlation = measurements.universal.constants.CORRELATION_DTYPE
        self.matrix_format_correlation = measurements.universal.constants.CORRELATION_FORMAT
    

    ## general sample data
    
    @property
    def sample_lsm(self):
        return self._sample_lsm
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
    def _sample_mean_and_deviation(self):
        return measurements.universal.sample_data.SampleMeanAndDeviation(self.points, self.values, self.sample_lsm)

    
    ## mean
    
    @property
    def sample_means(self):
        return self._sample_mean_and_deviation.sample_concentration_means(min_measurements=self.min_measurements_mean)

    @property
    @overrides.overrides
    def means(self):
        data = self.sample_means
        if data.count() == len(data):
            return data.data
        else:
            raise TooFewValuesError('It was not possible to calculate all values from the sample values, because to few sample values are available.')
    

    ## deviation
    
    @property
    def sample_concentration_standard_deviations(self):
        return self._sample_mean_and_deviation.sample_concentration_standard_deviations(min_measurements=self.min_measurements_standard_deviation, min_value=0)

    @property
    def concentration_standard_deviations(self):
        data = self.sample_concentration_standard_deviations
        if data.count() == len(data):
            return data.data
        else:
            raise TooFewValuesError('It was not possible to calculate all values from the sample values, because to few sample values are available.')


    @property
    def sample_average_noise_standard_deviations(self):
        return self._sample_mean_and_deviation.sample_average_noise_standard_deviations(min_measurements=self.min_measurements_standard_deviation, min_value=self.min_standard_deviation)

    @property
    def average_noise_standard_deviations(self):
        data = self.sample_average_noise_standard_deviations
        if data.count() == len(data):
            return data.data
        else:
            raise TooFewValuesError('It was not possible to calculate all values from the sample values, because to few sample values are available.')


    @property
    def sample_noise_standard_deviations(self):
        return self._sample_mean_and_deviation.sample_noise_standard_deviations(min_measurements=self.min_measurements_standard_deviation, min_value=self.min_standard_deviation)

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
        standard_deviations = (self.concentration_standard_deviations**2 + self.noise_standard_deviations**2)**(1/2)
        return standard_deviations
    

    ## correlation

    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.min_measurements_correlation', 'self.min_abs_correlation', 'self.max_abs_correlation', 'self.matrix_format_correlation', 'self.dtype_correlation'))
    def _sample_correlation(self):
        return measurements.universal.sample_data.SampleCorrelationMatrix(self, self.sample_lsm, self.min_measurements_correlation, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, matrix_format=self.matrix_format_correlation, dtype=self.dtype_correlation)
    

    @property
    def correlations_own_sample_matrix(self):
        return self._sample_correlation.correlation_matrix
    
    
    @property
    @overrides.overrides
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation)
        return correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)


    @property
    @overrides.overrides
    def correlations_own_cholesky_decomposition(self):
        import util.math.sparse.decompose.with_cholmod
        P, L = util.math.sparse.decompose.with_cholmod.cholesky(self.correlations_own, ordering_method=self.cholesky_ordering_method_correlation, return_type=util.math.sparse.decompose.with_cholmod.RETURN_P_L)
        return {'P': P.asformat(self.matrix_format_correlation).astype(np.int8), 'L': L.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)}




class MeasurementsAnnualPeriodic(MeasurementsAnnualPeriodicBase):
    
    def __init__(self, *args, **kargs):
        
        super().__init__(*args, **kargs)
        
        self.POSSIBLE_FILL_STRATEGIES = ('auto', 'point_average', 'lsm_average', 'constant', 'interpolate')
        self.POSSIBLE_KINDS = ('concentration_means', 'concentration_standard_deviations', 'average_noise_standard_deviations')
        self._interpolator_options = {}
        self._constant_fill_values = {}
    
    
    ## getter und setter
    
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
        
    
    def _check_kind(self, kind):
        if kind not in self.POSSIBLE_KINDS:
            raise ValueError('Kind must be in {}, but it is {}.'.format(self.POSSIBLE_KINDS, kind))
        return kind
    
    
    def get_interpolator_options(self, kind):
        try:
            return self._interpolator_options[kind]
        except KeyError:
            return (1,1,0,0)
    
    def set_interpolator_options(self, kind, value):
        self._interpolator_options[self._check_kind(kind)] = value    
    
    def get_constant_fill_value(self, kind):
        try:
            return self._constant_fill_values[kind]
        except KeyError:
            raise ValueError('Constant fill value is not set for kind {}.'.format(kind))
    
    def set_constant_fill_value(self, kind, value):
        self._constant_fill_values[self._check_kind(kind)] = value
    
    @property
    def interpolator_scaling_values(self):
        try:
            return self._interpolator_scaling_values
        except AttributeError:
            return measurements.universal.interpolate.default_scaling_values(self.sample_lsm)
    
    @interpolator_scaling_values.setter
    def interpolator_scaling_values(self, value):
        self._interpolator_scaling_values = value
    

    ## interpolater

    @property
    def _interpolator(self):
        return measurements.universal.interpolate.Interpolator_Annual_Periodic(self.sample_lsm, scaling_values=self.interpolator_scaling_values)
    

    def _choose_fill_strategy(self, number_of_sample_values):
        number_of_lsm_values = self.sample_lsm.number_of_map_indices
        sample_data_fill_amount = number_of_sample_values / number_of_lsm_values
        if sample_data_fill_amount >= 0.001:
            fill_strategy = 'interpolate'
        elif number_of_sample_values >= 1:
            fill_strategy = 'point_average'
        else:
            fill_strategy = 'constant'
        
        logger.debug('{}: Chosen {} as fill strategy, since {:d} sample data are for {:%} of the sampe lsm available.'.format(self.__class__.__name__, fill_strategy, number_of_sample_values, sample_data_fill_amount))
        return fill_strategy
        

    def _fill_strategy_for_kind(self, kind):
        ## choose fill method
        fill_strategy = self.fill_strategy
        if fill_strategy == 'auto':
            number_of_sample_values = len(self._data_map_indices_dict(kind))
            fill_strategy = self._choose_fill_strategy(number_of_sample_values)
        
        ## return
        logger.debug('{}: Fill startegy to use is {}.'.format(self.__class__.__name__, fill_strategy))
        return fill_strategy


    def _fill_strategy_with_number_of_sample_values(self, number_of_sample_values):
        ## choose fill method
        fill_strategy = self.fill_strategy
        if fill_strategy == 'auto':
            fill_strategy = self._choose_fill_strategy(number_of_sample_values)
        
        ## check number of available sample values
        if number_of_sample_values == 0 and fill_strategy != 'constant':
            raise TooFewValuesError('No sample values are available. Fill method {} is not applicable.'.format(fill_strategy))
        
        ## return
        logger.debug('{}: Fill startegy to use is {}.'.format(self.__class__.__name__, fill_strategy))
        return fill_strategy


    ## data general
    
    def _data_map_indices_dict(self, kind):
        if kind == 'concentration_means':        
            data_map_indices_dict = self._sample_mean_and_deviation.sample_concentration_means_map_indices_dict(min_measurements=self.min_measurements_mean)
        elif kind == 'concentration_standard_deviations':
            data_map_indices_dict = self._sample_mean_and_deviation.sample_concentration_standard_deviations_map_indices_dict(min_measurements=self.min_measurements_standard_deviation, min_value=0)
        elif kind == 'average_noise_standard_deviations':
            data_map_indices_dict = self._sample_mean_and_deviation.sample_average_noise_standard_deviations_map_indices_dict(min_measurements=self.min_measurements_standard_deviation, min_value=self.min_standard_deviation)
        else:
            raise ValueError('Unknown kind {}.'.format(kind))
        return data_map_indices_dict


    ## data for sample lsm 
    
    def _data_for_sample_lsm(self, kind):
        logger.debug('{}: Calculating {} data for sample lsm.'.format(self.__class__.__name__, kind))
        
        ## get data
        data_map_indices_dict = self._data_map_indices_dict(kind)
        map_indices_and_values = data_map_indices_dict.toarray()
        
        ## choose fill strategy
        fill_strategy = self._fill_strategy_with_number_of_sample_values(len(map_indices_and_values))
        logger.debug('{}: Filling sample lsm values with fill strategy {}.'.format(self.__class__.__name__, fill_strategy))
            
        ## apply fill_strategy
        if fill_strategy in ('point_average', 'lsm_average', 'constant'):
            
            if fill_strategy == 'point_average':
                sample_values = self._sample_mean_and_deviation._convert_map_indices_dict_to_array_for_points(data_map_indices_dict, is_discard_year=True)
                fill_value = sample_values.mean()
            elif fill_strategy == 'lsm_average':
                fill_value = map_indices_and_values[:,-1].mean()
            elif fill_strategy == 'constant':
                fill_value = self.get_constant_fill_value(kind)
                
            lsm_values = self.sample_lsm.insert_index_values_in_map(map_indices_and_values, no_data_value=fill_value, skip_values_on_land=False)
        
        elif fill_strategy == 'interpolate':
            interpolator_options = self.get_interpolator_options(kind)
            lsm_values = self._interpolator.interpolate_data_for_sample_lsm_with_map_indices(map_indices_and_values, interpolator_options)

        else:
            raise ValueError('Unknown fill method {}.'.format(fill_strategy))

        return lsm_values
    
    @property
    def means_for_sample_lsm(self):
        return self._data_for_sample_lsm('means')

    @property
    def concentration_standard_deviations_for_sample_lsm(self):
        return self._data_for_sample_lsm('concentration_standard_deviations')
    
    @property
    def average_noise_standard_deviations_for_sample_lsm(self):
        return self._data_for_sample_lsm('average_noise_standard_deviations')

    @property
    def standard_deviations_for_sample_lsm(self):
        return (self.concentration_standard_deviations_for_sample_lsm**2 + self.average_noise_standard_deviations_for_sample_lsm**2)**(1/2)
    
    
    ## data for sample points
    
    def _data_for_sample_points(self, kind):
        logger.debug('{}: Calculating {} data for sample points.'.format(self.__class__.__name__, kind))
        
        ## get data
        data_map_indices_dict = self._data_map_indices_dict(kind)
        data = self._sample_mean_and_deviation._convert_map_indices_dict_to_array_for_points(data_map_indices_dict, is_discard_year=True)
        number_of_values = data.count()
        number_of_points = len(data)

        logger.debug('{}: Got values for {:d} of {:d} points with sample data.'.format(self.__class__.__name__, number_of_values, number_of_points))
        
        ## fill if empty values
        if number_of_values < number_of_points:
            ## choose fill strategy
            fill_strategy = self._fill_strategy_with_number_of_sample_values(len(data_map_indices_dict))
            logger.debug('{}: Filling remaining {:%} sample points values with fill strategy {}.'.format(self.__class__.__name__, 1-number_of_values/number_of_points, fill_strategy))
            
            ## fill
            if fill_strategy == 'point_average':
                data[data.mask] = data.mean()
            elif fill_strategy == 'lsm_average':
                data[data.mask] = data_map_indices_dict.values().mean()
            elif fill_strategy == 'interpolate':
                data[data.mask] = self._interpolator.interpolate_data_for_points_from_interpolated_lsm_data(self._data_for_sample_lsm(kind), self.points[data.mask])
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

    


class MeasurementsAnnualPeriodicCache(MeasurementsAnnualPeriodic):
    
    ## points and values cache files

    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def points(self):
        return super().points
    
    def points_cache_file(self):
        return measurements.universal.constants.POINTS_FILE.format(tracer=self.tracer, data_set=self.data_set_name)


    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def values(self):
        return super().values
    
    def values_cache_file(self):
        return measurements.universal.constants.VALUES_FILE.format(tracer=self.tracer, data_set=self.data_set_name)


    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def measurements_dict(self):
        return super().measurements_dict    
    
    def measurements_dict_cache_file(self):
        return measurements.universal.constants.MEASUREMENTS_DICT_FILE.format(tracer=self.tracer, data_set=self.data_set_name)
    
    
    ## means
    
    def _fill_strategy_str(self, kind):
        ## if standrad deviation, use str for concentration and average_noise
        if kind == 'standard_deviations':
            concentration_fill_strategy = self._fill_strategy_str('concentration_standard_deviations')
            average_noise_fill_strategy = self._fill_strategy_str('average_noise_standard_deviations')
            ## if same for both, use only once
            if concentration_fill_strategy == average_noise_fill_strategy:
                fill_strategy = concentration_fill_strategy
            ## if both use interpolation, use only once but two interpolator_options
            elif concentration_fill_strategy.startswith('interpolate') and average_noise_fill_strategy.startswith('interpolate'):
                interpolator_options = '+'.join([','.join(map(str, self.get_interpolator_options(kind))) for kind in ('concentration_standard_deviations', 'average_noise_standard_deviations')])
                fill_strategy = measurements.universal.constants.INTERPOLATION_FILL_STRATEGY.format(scaling_values=','.join(map(str, self.interpolator_scaling_values)), interpolator_options=interpolator_options)
            ## if different strategies, append both strategies
            else:
                fill_strategy = concentration_fill_strategy + '_+_' + average_noise_fill_strategy
        ## else, get used fill strategy
        else:
            if kind == 'noise_standard_deviations':
                fill_strategy = self._fill_strategy_for_kind('average_' + kind)
            else:
                fill_strategy = self._fill_strategy_for_kind(kind)
        
            ## if interpolation, append options for interpolations
            if fill_strategy == 'interpolate':
                fill_strategy = measurements.universal.constants.INTERPOLATION_FILL_STRATEGY.format(scaling_values=','.join(map(str, self.interpolator_scaling_values)), interpolator_options=','.join(map(str, self.get_interpolator_options(kind))))
            
        return fill_strategy
    
    
    def _mean_cache_file(self, target):
        fill_strategy=self._fill_strategy_str('concentration_means')
        return measurements.universal.constants.MEAN_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_mean, fill_strategy=fill_strategy, target=target)   
    
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_mean'))
    @util.cache.file.decorator()
    @overrides.overrides
    def means(self):
        return super().means
    
    def means_cache_file(self):
        return self._mean_cache_file('sample_points')    
    

    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_mean'))
    @util.cache.file.decorator()
    @overrides.overrides
    def means_for_sample_lsm(self):
        return super().means_for_sample_lsm
    
    def means_for_sample_lsm_cache_file(self):
        return self._mean_cache_file(str(self.sample_lsm))
    
    
    ## deviation
    
    def _standard_deviations_cache_file(self, deviation_type, target):
        fill_strategy = self._fill_strategy_str(deviation_type)
        if deviation_type == 'concentration_standard_deviations':
            min_standard_deviation = 0
        else:
            min_standard_deviation = self.min_standard_deviation
        return measurements.universal.constants.DEVIATION_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements_standard_deviation, min_standard_deviation=min_standard_deviation, deviation_type=deviation_type, fill_strategy=fill_strategy, target=target)
    
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def concentration_standard_deviations(self):
        return super().concentration_standard_deviations
    
    def concentration_standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('concentration_standard_deviations', 'sample_points')
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def concentration_standard_deviations_for_sample_lsm(self):
        return super().concentration_standard_deviations_for_sample_lsm
    
    def concentration_standard_deviations_for_sample_lsm_cache_file(self):
        return self._standard_deviations_cache_file('concentration_standard_deviations', str(self.sample_lsm))
    
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def average_noise_standard_deviations(self):
        return super().average_noise_standard_deviations
    
    def average_noise_standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('average_noise_standard_deviations', 'sample_points')    
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def average_noise_standard_deviations_for_sample_lsm(self):
        return super().average_noise_standard_deviations_for_sample_lsm
    
    def average_noise_standard_deviations_for_sample_lsm_cache_file(self):
        return self._standard_deviations_cache_file('average_noise_standard_deviations', str(self.sample_lsm))


    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def noise_standard_deviations(self):
        return super().noise_standard_deviations
    
    def noise_standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('noise_standard_deviations', 'sample_points')
    
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def standard_deviations(self):
        return super().standard_deviations
    
    def standard_deviations_cache_file(self):
        return self._standard_deviations_cache_file('standard_deviations', 'sample_points')
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def standard_deviations_for_sample_lsm(self):
        return super().standard_deviations_for_sample_lsm
    
    def standard_deviations_for_sample_lsm_cache_file(self):
        return self._standard_deviations_cache_file('standard_deviations', str(self.sample_lsm))
    
    
    ## correlation
    
    @property
    @overrides.overrides
    def _sample_correlation(self):
        return measurements.universal.sample_data.SampleCorrelationMatrixCache(self, self.sample_lsm, self.min_measurements_correlation, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, matrix_format=self.matrix_format_correlation, dtype=self.dtype_correlation)

    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation', 'self.min_measurements_correlation', 'self.min_abs_correlation', 'self.max_abs_correlation', 'self.cholesky_min_diag_value_correlation', 'self.cholesky_ordering_method_correlation', 'self.cholesky_reordering_correlation', 'self.matrix_format_correlation', 'self.dtype_correlation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation, reduction_factors_file=self.reduction_factors_cache_file())
        return correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)
    
    def correlations_own_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements_standard_deviation=self.min_measurements_standard_deviation, min_standard_deviation=self.min_standard_deviation, fill_strategy=self._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.CORRELATION_MATRIX_POSITIVE_DEFINITE_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements_correlation, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, standard_deviation_description=standard_deviation_description, dtype=self.dtype_correlation, matrix_format=self.matrix_format_correlation)
    
    def reduction_factors_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements_standard_deviation=self.min_measurements_standard_deviation, min_standard_deviation=self.min_standard_deviation, fill_strategy=self._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements_correlation, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, standard_deviation_description=standard_deviation_description) 

    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation', 'self.min_measurements_correlation', 'self.min_abs_correlation', 'self.max_abs_correlation', 'self.cholesky_min_diag_value_correlation', 'self.cholesky_ordering_method_correlation', 'self.cholesky_reordering_correlation', 'self.matrix_format_correlation', 'self.dtype_correlation'))
    def _correlations_own_cholesky_decomposition(self):
        return super().correlations_own_cholesky_decomposition
    
    @property
    @util.cache.file.decorator()
    def _correlations_own_cholesky_decomposition_P(self):
        return self._correlations_own_cholesky_decomposition['P']
    
    def _correlations_own_cholesky_decomposition_P_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements_standard_deviation=self.min_measurements_standard_deviation, min_standard_deviation=self.min_standard_deviation, fill_strategy=self._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements_correlation, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, standard_deviation_description=standard_deviation_description, dtype=np.int8, matrix_format=self.matrix_format_correlation, factor_type='P')
    
    @property
    @util.cache.file.decorator()
    def _correlations_own_cholesky_decomposition_L(self):
        return self._correlations_own_cholesky_decomposition['L']
    
    def _correlations_own_cholesky_decomposition_L_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements_standard_deviation=self.min_measurements_standard_deviation, min_standard_deviation=self.min_standard_deviation, fill_strategy=self._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.CORRELATION_MATRIX_CHOLESKY_FACTOR_FILENAME.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements_correlation, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, ordering_method=self.cholesky_ordering_method_correlation, reordering=self.cholesky_reordering_correlation, cholesky_min_diag_value=self.cholesky_min_diag_value_correlation, standard_deviation_description=standard_deviation_description, dtype=self.dtype_correlation, matrix_format=self.matrix_format_correlation, factor_type='L')
    
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name', 'self.fill_strategy', 'self.min_measurements_standard_deviation', 'self.min_standard_deviation', 'self.min_measurements_correlation', 'self.min_abs_correlation', 'self.max_abs_correlation', 'self.cholesky_min_diag_value_correlation', 'self.cholesky_ordering_method_correlation', 'self.cholesky_reordering_correlation', 'self.matrix_format_correlation', 'self.dtype_correlation'))
    @overrides.overrides
    def correlations_own_cholesky_decomposition(self):
        return {'P': self._correlations_own_cholesky_decomposition_P, 'L': self._correlations_own_cholesky_decomposition_L}
    



## near water

class MeasurementsNearWater(Measurements):
    
    def __init__(self, base_measurements, water_lsm=None, max_box_distance_to_water=0):
        self.base_measurements = base_measurements
        self.max_box_distance_to_water = max_box_distance_to_water
        
        if water_lsm is None:
            water_lsm = self.base_measurements.sample_lsm
        water_lsm = water_lsm.copy()
        water_lsm.t_dim = 0
        self.water_lsm = water_lsm
        
        Measurements.__init__(self)

    
    ## properties
    @property
    def tracer(self):
        return self.base_measurements.tracer
    
    @property
    def data_set_name(self):
        return measurements.universal.constants.NEAR_WATER_DATA_SET_NAME.format(base_data_set_name=self.base_measurements.data_set_name, water_lsm=self.water_lsm, max_box_distance_to_water=self.max_box_distance_to_water)
    
    
    ## projection methods
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
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
    @overrides.overrides
    def points(self):
        return self.near_water_projection_matrix * self.base_measurements.points

    @property
    @overrides.overrides
    def values(self):
        return self.near_water_projection_matrix * self.base_measurements.values
    
    @property
    @overrides.overrides
    def means(self):
        return self.near_water_projection_matrix * self.base_measurements.means

    @property
    @overrides.overrides
    def standard_deviations(self):
        return self.near_water_projection_matrix * self.base_measurements.standard_deviations
    
    @property
    @overrides.overrides
    def correlations_own(self):
        return self.near_water_projection_matrix * self.base_measurements.correlations_own * self.near_water_projection_matrix.T

    @overrides.overrides
    def correlations_other(self, measurements=None):
        return self.near_water_projection_matrix * self.base_measurements.correlations_other(measurements=measurements)



class MeasurementsAnnualPeriodicNearWater(MeasurementsNearWater, MeasurementsAnnualPeriodic):
    
    def __init__(self, base_measurements, water_lsm=None, max_box_distance_to_water=0):
        MeasurementsAnnualPeriodicBase.__init__(self, base_measurements.sample_lsm, min_standard_deviation=base_measurements.min_standard_deviation, min_abs_correlation=base_measurements.min_abs_correlation, max_abs_correlation=base_measurements.max_abs_correlation, min_measurements_mean=base_measurements.min_measurements_mean, min_measurements_standard_deviation=base_measurements.min_measurements_standard_deviation, min_measurements_correlation=base_measurements.min_measurements_correlation)
        super().__init__(base_measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)
    
    
    @property
    @overrides.overrides
    def concentration_standard_deviations(self):
        return self.near_water_projection_matrix * self.base_measurements.concentration_standard_deviations

    @property
    @overrides.overrides
    def noise_standard_deviations(self):
        return self.near_water_projection_matrix * self.base_measurements.noise_standard_deviations

    @property
    @overrides.overrides
    def average_noise_standard_deviations(self):
        return self.near_water_projection_matrix * self.base_measurements.average_noise_standard_deviations
    
    
    @property
    @overrides.overrides
    def correlations_own_sample_matrix(self):
        return self.near_water_projection_matrix * self.base_measurements.correlations_own_sample_matrix * self.near_water_projection_matrix.T
    
    @property
    @overrides.overrides
    def correlations_own(self):
        return MeasurementsAnnualPeriodicBase.correlations_own
        


class MeasurementsAnnualPeriodicNearWaterCache(MeasurementsAnnualPeriodicCache, MeasurementsAnnualPeriodicNearWater):
    
    ## cacheable properties
    
    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def near_water_projection_matrix(self):
        return super().near_water_projection_matrix
    

    @property
    @util.cache.memory.method_decorator(dependency=('self.tracer', 'self.data_set_name'))
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own_sample_matrix(self):
        return super().correlations_own_sample_matrix
    

    ## cache files

    def near_water_projection_matrix_cache_file(self):
        return measurements.universal.constants.NEAR_WATER_PROJECTION_MASK_FILE.format(tracer=self.tracer, data_set=self.data_set_name, sample_lsm=self.sample_lsm, water_lsm=self.water_lsm, max_box_distance_to_water=self.max_box_distance_to_water, matrix_format='csc')

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
        return self.base_measurements._sample_correlation.correlation_matrix_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def correlations_own_cache_file(self):
        return self.base_measurements.correlations_own_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def reduction_factors_cache_file(self):
        return self.base_measurements.reduction_factors_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def _correlations_own_cholesky_decomposition_P_cache_file(self):
        return self.base_measurements._correlations_own_cholesky_decomposition_P_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)

    @overrides.overrides
    def _correlations_own_cholesky_decomposition_L_cache_file(self):
        return self.base_measurements._correlations_own_cholesky_decomposition_L_cache_file().replace(self.base_measurements.data_set_name, self.data_set_name)




## union

class MeasurementsAnnualPeriodicUnion(MeasurementsAnnualPeriodic):
    
    def __init__(self, *measurements_list):
        logger.debug('Initiating {} with measurements {}.'.format(self.__class__.__name__, measurements_list))
        
        if len(measurements_list) == 0:
            raise ValueError('{} must be initiated with at least one measurement object.'.format(self.__class__.__name__))
        
        ## sort
        measurements_list = sorted(measurements_list, key=lambda measurement: measurement.data_set_name)
        
        ## check same tracer and data set name
        n = len(measurements_list)
        for i in range(n-1):
            if measurements_list[i].tracer != measurements_list[i+1].tracer:
                raise ValueError('Measurements objects with different tracers ({} and {}) are not allowed!'.format(measurements_list[i].tracer, measurements_list[i+1].tracer))
            if measurements_list[i].data_set_name == measurements_list[i+1].data_set_name:
                raise ValueError('Measurements objects with same tracer ({}) and same data set name ({}) are not allowed!'.format(measurements_list[i].tracer, measurements_list[i].data_set_name))
            if measurements_list[i].sample_lsm != measurements_list[i+1].sample_lsm:
                raise ValueError('Measurements objects with different sample lsm ({} and {}) are not allowed!'.format(measurements_list[i].sample_lsm, measurements_list[i+1].sample_lsm))

        ## store
        self.measurements_list = measurements_list
        
        ## chose values for union
        sample_lsm = measurements_list[0].sample_lsm
        tracer = measurements_list[0].tracer
        data_set_name = ','.join(map(lambda measurement: measurement.data_set_name, measurements_list))
        
        min_standard_deviation = min(map(lambda measurement: measurement.min_standard_deviation, measurements_list))
        min_abs_correlation = min(map(lambda measurement: measurement.min_abs_correlation, measurements_list))
        max_abs_correlation = max(map(lambda measurement: measurement.max_abs_correlation, measurements_list))
        min_measurements_mean = min(map(lambda measurement: measurement.min_measurements_mean, measurements_list))
        min_measurements_standard_deviation = min(map(lambda measurement: measurement.min_measurements_standard_deviation, measurements_list))
        min_measurements_correlation = min(map(lambda measurement: measurement.min_measurements_correlation, measurements_list))
        
        ## call super init
        super().__init__(sample_lsm, tracer=tracer, data_set_name=data_set_name, min_standard_deviation=min_standard_deviation, min_abs_correlation=min_abs_correlation, max_abs_correlation=max_abs_correlation, min_measurements_mean=min_measurements_mean, min_measurements_standard_deviation=min_measurements_standard_deviation, min_measurements_correlation=min_measurements_correlation)
        
    
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



class MeasurementsAnnualPeriodicUnionCache(MeasurementsAnnualPeriodicUnion, MeasurementsAnnualPeriodicCache):
    pass
    



## collection

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
        self._measurements_list = measurements_list
        
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
    
    
    @property
    def measurements_list(self):
        return self._measurements_list
    
    
    def __str__(self):
        return ','.join(map(str, self.measurements_list))
    
    
    def __iter__(self):
        return self.measurements_list.__iter__()
    
    
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
    def correlations_own_sample_matrix(self):
        n = len(self.measurements_list)
        correlations = np.array([n,n], dtype=object)
        
        for i in range(n):
            measurements_i = self.measurements_list[i]
            correlations[i, i] = measurements_i.correlations()
            for j in range(i+1, n):
                measurements_j = self.measurements_list[j]
                correlations[i, j] = measurements_i.correlations(measurements_j)
                correlations[j, i] = correlations[i, j].T
        
        correlations = scipy.sparse.bmat(correlations, format=self.matrix_format_correlation, dtype=self.dtype_correlation)
        assert correlations.shape == (self.number_of_measurements, self.number_of_measurements)
        return correlations

    
    @property
    @overrides.overrides
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation)
        correlation_matrix = correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)
        assert correlation_matrix.shape == (self.number_of_measurements, self.number_of_measurements)
        return correlation_matrix


    @property
    @overrides.overrides
    def correlations_own_cholesky_decomposition(self):
        import util.math.sparse.decompose.with_cholmod
        P, L = util.math.sparse.decompose.with_cholmod.cholesky(self.correlations_own, ordering_method=self.cholesky_ordering_method_correlation, return_type=util.math.sparse.decompose.with_cholmod.RETURN_P_L)
        return {'P': P.asformat(self.matrix_format_correlation).astype(np.int8), 'L': L.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)}


    def _measurements_dict(self, convert_function=None):
        if convert_function is None:
            convert_function = lambda x: x
        
        results = {}
        
        for measurement in self.measurements_list:
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
        value_list = [measurements_dict[measurement.tracer][measurement.data_set_name] for measurement in self.measurements_list]
        return np.concatenate(value_list)
    
    
    def subset(self, tracers):
        measurements_list = [measurement for measurement in self.measurements_list if measurement.tracer in tracers]
        subset = type(self)(*measurements_list)
        return subset
        



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
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own_sample_matrix(self):
        return super().correlations_own_sample_matrix

    @property
    @util.cache.file.decorator()
    @overrides.overrides
    def correlations_own(self):
        import util.math.sparse.decompose.with_cholmod
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlations_own_sample_matrix, min_abs_value=self.min_abs_correlation, min_diag_value=self.cholesky_min_diag_value_correlation, ordering_method=self.cholesky_ordering_method_correlation, reorder_after_each_step=self.cholesky_reordering_correlation, reduction_factors_file=self.reduction_factors_file)
        return correlation_matrix.asformat(self.matrix_format_correlation).astype(self.dtype_correlation)

    @property
    @util.cache.memory.method_decorator()
    def _correlations_own_cholesky_decomposition(self):
        return super().correlations_own_cholesky_decomposition
    
    @property
    @util.cache.file.decorator()
    def _correlations_own_cholesky_decomposition_P(self):
        return self._correlations_own_cholesky_decomposition['P']
    
    @property
    @util.cache.file.decorator()
    def _correlations_own_cholesky_decomposition_L(self):
        return self._correlations_own_cholesky_decomposition['L']
    
    @property
    @overrides.overrides
    def correlations_own_cholesky_decomposition(self):
        return {'P': self._correlations_own_cholesky_decomposition_P, 'L': self._correlations_own_cholesky_decomposition_L}
    

    def reduction_factors_cache_file(self):
        return self._merge_files([measurement.reduction_factors_cache_file for measurement in self.measurements_list])

    def correlations_own_cache_file(self):
        return self._merge_files([measurement.correlations_own_cache_file for measurement in self.measurements_list])

    def _correlations_own_cholesky_decomposition_P_cache_file(self):
        return self._merge_files([measurement._correlations_own_cholesky_decomposition_P_cache_file for measurement in self.measurements_list])

    def _correlations_own_cholesky_decomposition_L_cache_file(self):
        return self._merge_files([measurement._correlations_own_cholesky_decomposition_L_cache_file for measurement in self.measurements_list])




def as_measurements_collection(measurements):
    if isinstance(measurements, MeasurementsCollectionCache):
        return measurements
    else:
        return MeasurementsCollection(*measurements)
    
