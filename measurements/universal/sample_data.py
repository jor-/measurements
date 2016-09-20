import numpy as np
import scipy.sparse

import util.math.util
import util.cache.file_based
import util.logging

import measurements.universal.dict
import measurements.universal.constants
import measurements.constants

logger = util.logging.logger



class SampleMeanAndDeviation():
    
    def __init__(self, points, values, sample_lsm):
        self.points = points
        self.values = values
        self.sample_lsm = sample_lsm

    @property
    def measurement_dict(self):
        measurement_dict = measurements.universal.dict.MeasurementsDict()
        measurement_dict.append_values(self.points, self.values)
        return measurement_dict

    
    ## general
    
    def _convert_map_indices_dict_to_array_for_points(self, data_dict, is_discard_year):
        points = self.sample_lsm.coordinates_to_map_indices(self.points, discard_year=is_discard_year, int_indices=True)
        n = len(points)
        point_data = np.ma.masked_all(n)
        for i in range(n):
            value_list_i = data_dict[points[i]]
            assert len(value_list_i) in [0, 1]
            if len(value_list_i) == 1:
                point_data[i] = value_list_i[0]
        return point_data


    def _convert_map_indices_dict_to_points_and_values_array(self, data_dict):
        data_dict.map_indices_to_coordinates(self.sample_lsm)
        return data_dict.toarray()


    def _sample_data_dict_concentration_based(self, data_function):
        data_dict = self.measurement_dict
        data_dict.coordinates_to_map_indices(self.sample_lsm, int_indices=True)
        data_dict.means(min_number_of_values=1, return_type='self')
        data_dict.discard_year()
        data_function(data_dict)
        return data_dict
    
    def _sample_data_dict_average_noise_based(self, data_function, return_values_at_points=True):
        data_dict = self.measurement_dict
        data_dict.coordinates_to_map_indices(self.sample_lsm, int_indices=True)
        data_function(data_dict)
        data_dict.discard_year()
        data_dict.means(min_number_of_values=1, return_type='self')
        return data_dict


    def _sample_data_dict_noise_based(self, data_function, return_values_at_points=True):
        data_dict = self.measurement_dict
        data_dict.coordinates_to_map_indices(self.sample_lsm, int_indices=True)
        data_function(data_dict)
        return data_dict

    
    ## mean

    def sample_concentration_means_map_indices_dict(self, min_measurements=measurements.constants.MEAN_MIN_MEASUREMENTS):
        logger.debug('Calculating sample_concentration_means_map_indices_dict with min_measurements {}.'.format(min_measurements))
        data_function = lambda data_dict: data_dict.means(min_number_of_values=min_measurements, return_type='self')
        data = self._sample_data_dict_concentration_based(data_function)
        return data

    def sample_concentration_means(self, min_measurements=measurements.constants.MEAN_MIN_MEASUREMENTS):
        logger.debug('Calculating sample_concentration_means with min_measurements {}.'.format(min_measurements))
        data_dict = self.sample_concentration_means_map_indices_dict(min_measurements=min_measurements)
        data = self._convert_map_indices_dict_to_array_for_points(data_dict, is_discard_year=True)
        return data
    

    ## deviation
    
    def sample_concentration_standard_deviations_map_indices_dict(self, min_measurements=measurements.constants.DEVIATION_MIN_MEASUREMENTS, min_value=0):
        logger.debug('Calculating sample_concentration_standard_deviations with min_measurements {} and min_value {}.'.format(min_measurements, min_value))
        data_function = lambda data_dict: data_dict.standard_deviations(min_number_of_values=min_measurements, min_value=min_value, return_type='self')
        data_dict = self._sample_data_dict_concentration_based(data_function) 
        return data_dict

    
    def sample_concentration_standard_deviations(self, min_measurements=measurements.constants.DEVIATION_MIN_MEASUREMENTS, min_value=0):
        logger.debug('Calculating sample_concentration_standard_deviations with min_measurements {} and min_value {}.'.format(min_measurements, min_value))
        data_dict = self.sample_concentration_standard_deviations_map_indices_dict(min_measurements=min_measurements, min_value=min_value)
        data = self._convert_map_indices_dict_to_array_for_points(data_dict, is_discard_year=True)
        return data


    def sample_average_noise_standard_deviations_map_indices_dict(self, min_measurements=measurements.constants.DEVIATION_MIN_MEASUREMENTS, min_value=0):
        logger.debug('Calculating sample_average_noise_standard_deviations with min_measurements {} and min_value {}.'.format(min_measurements, min_value))
        data_function = lambda data_dict: data_dict.standard_deviations(min_number_of_values=min_measurements, min_value=min_value, return_type='self')
        data_dict = self._sample_data_dict_average_noise_based(data_function)
        return data_dict


    def sample_average_noise_standard_deviations(self, min_measurements=measurements.constants.DEVIATION_MIN_MEASUREMENTS, min_value=0):
        logger.debug('Calculating sample_average_noise_standard_deviations with min_measurements {} and min_value {}.'.format(min_measurements, min_value))
        data_dict = self.sample_average_noise_standard_deviations_map_indices_dict(min_measurements=min_measurements, min_value=min_value)
        data = self._convert_map_indices_dict_to_array_for_points(data_dict, is_discard_year=True)
        return data

    
    def sample_noise_standard_deviations_map_indices_dict(self, min_measurements=measurements.constants.DEVIATION_MIN_MEASUREMENTS, min_value=0):
        logger.debug('Calculating sample_noise_standard_deviations with min_measurements {} and min_value {}.'.format(min_measurements, min_value))
        data_function = lambda data_dict: data_dict.standard_deviations(min_number_of_values=min_measurements, min_value=min_value, return_type='self')
        data_dict = self._sample_data_dict_noise_based(data_function)
        return data_dict

    
    def sample_noise_standard_deviations(self, min_measurements=measurements.constants.DEVIATION_MIN_MEASUREMENTS, min_value=0):
        logger.debug('Calculating sample_noise_standard_deviations with min_measurements {} and min_value {}.'.format(min_measurements, min_value))
        data_dict = self.sample_noise_standard_deviations_map_indices_dict(min_measurements=min_measurements, min_value=min_value)
        data = self._convert_map_indices_dict_to_array_for_points(data_dict, is_discard_year=False)
        return data
    
    

class SampleCorrelationMatrix:
    
    def __init__(self, measurements, sample_lsm, min_measurements, max_year_diff=float('inf'), min_abs_correlation=measurements.universal.constants.CORRELATION_MIN_ABS_VALUE, max_abs_correlation=measurements.universal.constants.CORRELATION_MAX_ABS_VALUE, dtype=np.float32, matrix_format='csc'):
        
        self.measurements = measurements        
        self.sample_lsm = sample_lsm

        self.min_measurements = min_measurements
        self.max_year_diff = max_year_diff
        self.min_abs_correlation = min_abs_correlation
        self.max_abs_correlation = max_abs_correlation

        self.dtype = np.dtype(dtype)
        self.matrix_format = matrix_format


    ## properties

    @property
    def shape(self):
        n = len(self.measurements.points)
        return (n, n)


    ## standard deviation

    @property
    def standard_deviations(self):
        standard_deviations = self.measurements.standard_deviations
        assert np.all(standard_deviations > 0)
        return standard_deviations
    

    ##  map index to point index dict

    def map_indices_to_point_index_dict(self, discard_year=False):
        logger.debug('Calculating map index to point index dict with discard year {}.'.format(discard_year))

        points = self.measurements.points
        map_indices_to_point_index_dict = measurements.universal.dict.MeasurementsDict()

        for i in range(len(points)):
            point = points[i]
            map_index = self.sample_lsm.coordinates_to_map_indices(point, discard_year=discard_year, int_indices=True)
            map_indices_to_point_index_dict.append_value(map_index, i)

        return map_indices_to_point_index_dict
    

    ## same box correlation
    
    @property
    def same_box_correlation_matrix_lower_triangle(self):
        logger.debug('Calculating same box correlation matrix lower triangle with minimal absolute correlation {} in matrix format {} with dtype {}.'.format(self.min_abs_correlation, self.matrix_format, self.dtype))
        
        ## create lil matrix
        correlation_matrix = scipy.sparse.lil_matrix(self.shape, dtype=self.dtype)
        
        ## get values
        standard_deviations = self.standard_deviations
        map_indices_to_point_index_dict = self.map_indices_to_point_index_dict(discard_year=False)
        same_box_sample_covariance_index_array = self.measurements.concentration_standard_deviations**2
    
        ## iterate all point pairs in same box
        for key, point_index_list in map_indices_to_point_index_dict.iterator_keys_and_value_lists():
            n = len(point_index_list)
            
            for i in range(n):
                point_index_i = point_index_list[i]
                
                covariance = same_box_sample_covariance_index_array[point_index_i]
                
                for j in range(i+1, n):
                    point_index_j = point_index_list[j]
                    
                    ## calculate correlation
                    assert same_box_sample_covariance_index_array[point_index_i] == same_box_sample_covariance_index_array[point_index_j]
                    correlation = covariance / (standard_deviations[point_index_i] * standard_deviations[point_index_j])
                    assert correlation >= 0 and correlation <= 1

                    ## if abs correlation geq min correlation, insert 
                    if np.abs(correlation) >= self.min_abs_correlation:

                        point_index_min, point_index_max = (min(point_index_i, point_index_j), max(point_index_i, point_index_j))
                        assert point_index_min < point_index_max
                        correlation_matrix[point_index_max, point_index_min] = correlation
                        logger.debug('Same box correlation {} added to same box lower triangle matrix at ({}, {}).'.format(correlation, point_index_max, point_index_min))

            
        ## convert to needed format
        if self.matrix_format != 'lil':
            logger.debug('Converting matrix to format {} and dtype {}.'.format(self.matrix_format, self.dtype))
            correlation_matrix = correlation_matrix.asformat(self.matrix_format).astype(self.dtype)

        logger.debug('Calculated same box correlation lower triangle matrix with {} entries for minimal absolute correlation {} in matrix format {} with dtype {}.'.format(correlation_matrix.nnz, self.min_abs_correlation, self.matrix_format, self.dtype))
        return correlation_matrix

    
    ## different_boxes_sample_covariances
    
    @property
    def concentrations_same_points_except_year_dict(self):
        m = self.measurements.measurements_dict
        m.categorize_indices_to_lsm(self.sample_lsm, discard_year=False)
        m.means(return_type='self')
        return m.filter_same_points_except_year(min_number_of_values=self.min_measurements)

    
    @property
    def sample_covariance_dict(self):
        ms = self.concentrations_same_points_except_year_dict
        covariance = ms.correlation_or_covariance('covariance', min_number_of_values=self.min_measurements, stationary=False, max_year_diff=self.max_year_diff)
        return covariance
    
    
    def different_boxes_sample_covariances_iterator(self):
        different_boxes_sample_covariances_dict = self.sample_covariance_dict
        different_boxes_sample_covariances_dict.coordinates_to_map_indices(self.sample_lsm, int_indices=True)
        for map_indices, value_list in different_boxes_sample_covariances_dict.iterator_keys_and_value_lists():
            assert len(map_indices) == 2
            assert len(value_list) == 1
            values = value_list[0]
            assert len(values) == 2
            quantity, covariance = values
            yield map_indices, quantity, covariance
    

    ## different boxes covariance and quantity matrix

    @property
    def different_boxes_quantity_lower_triangle_matrix(self):
        logger.debug('Calculating different boxes quantity lower triangle matrix with minimal absolute correlation {} in matrix format {}.'.format(self.min_abs_correlation, self.matrix_format))

        ## get max quantity
        max_quantity = 0
        for map_indices, quantity, covariance in self.different_boxes_sample_covariances_iterator():
            max_quantity = max(max_quantity, quantity)
        
        ## get matrix dtype
        dtype = util.math.util.min_int_dtype(max_quantity, unsigned=True)
        dtype = np.dtype(dtype)
        
        ## create lil matrix
        quantity_matrix = scipy.sparse.lil_matrix(self.shape, dtype=dtype)
        
        ## get values
        standard_deviations = self.standard_deviations
        points = self.measurements.points
        map_indices_to_point_index_dict = self.map_indices_to_point_index_dict(discard_year=False)
        map_indices_to_point_index_year_discarded_dict = self.map_indices_to_point_index_dict(discard_year=True)
    
        ## iterate over sample covariances
        for map_indices, quantity, covariance in self.different_boxes_sample_covariances_iterator():
            map_indices_array = np.array(map_indices)
            map_indices_diff = map_indices_array[1] - map_indices_array[0]
            logger.debug('For different box correlation matrix entires with map indices {} inserting quantity {} (covariance {}).'.format(map_indices, quantity, covariance))
            
            ## iterate over all index pairs with sample covariance
            for point_index_i in map_indices_to_point_index_year_discarded_dict[map_indices[0]]:
                point_i = points[point_index_i]
                point_i_map_index = self.sample_lsm.coordinate_to_map_index(*point_i, discard_year=False, int_indices=True)
                point_j_map_index = tuple(point_i_map_index + map_indices_diff)
                
                for point_index_j in map_indices_to_point_index_dict[point_j_map_index]:
                    
                    ## calculate correlation
                    correlation = covariance / (standard_deviations[point_index_i] * standard_deviations[point_index_j])
                    
                    ## insert quantity
                    if np.abs(correlation) >= self.min_abs_correlation:
                        point_index_min, point_index_max = (min(point_index_i, point_index_j), max(point_index_i, point_index_j))
                        assert point_index_min < point_index_max and point_index_min in (point_index_i, point_index_j) and point_index_max in (point_index_i, point_index_j)
                        quantity_matrix[point_index_max, point_index_min] = quantity

        if self.matrix_format != 'lil':
            logger.debug('Converting quantity matrix to format {}.'.format(self.matrix_format))
            if self.matrix_format == 'csc':
                quantity_matrix = quantity_matrix.asformat('csr')
                logger.debug('quantity matrix converted to format csr.')
            quantity_matrix = quantity_matrix.asformat(self.matrix_format).astype(dtype)
            logger.debug('quantity matrix converted to format {}.'.format(self.matrix_format))

        logger.debug('Calculated differend boxes quantity lower triangle matrix with {} entries for minimal absolute correlation {}.'.format(correlation_matrix.nnz, self.min_abs_correlation))

        return quantity_matrix
    
    
    @property
    def different_boxes_correlation_lower_triangle_matrix(self):
        logger.debug('Calculating different boxes correlation lower triangle matrix with minimal absolute correlation {} in matrix format {} with dtype {}.'.format(self.min_abs_correlation, self.matrix_format, self.dtype))
        
        ## create lil matrix
        correlation_matrix = scipy.sparse.lil_matrix(self.shape, dtype=self.dtype)
        
        ## get values
        standard_deviations = self.standard_deviations
        points = self.measurements.points
        map_indices_to_point_index_dict = self.map_indices_to_point_index_dict(discard_year=False)
        map_indices_to_point_index_year_discarded_dict = self.map_indices_to_point_index_dict(discard_year=True)
    
        ## iterate over sample covariances
        for map_indices, quantity, covariance in self.different_boxes_sample_covariances_iterator():
            map_indices_array = np.array(map_indices)
            map_indices_diff = map_indices_array[1] - map_indices_array[0]
            logger.debug('For different box correlation matrix entires with map indices {} inserting covariance {} (quantity {}).'.format(map_indices, covariance, quantity))
            
            ## iterate over all index pairs with sample covariance
            for point_index_i in map_indices_to_point_index_year_discarded_dict[map_indices[0]]:
                point_i = points[point_index_i]
                point_i_map_index = self.sample_lsm.coordinate_to_map_index(*point_i, discard_year=False, int_indices=True)
                point_j_map_index = tuple(point_i_map_index + map_indices_diff)
                
                for point_index_j in map_indices_to_point_index_dict[point_j_map_index]:
                    
                    ## calculate correlation
                    correlation = covariance / (standard_deviations[point_index_i] * standard_deviations[point_index_j])
                    
                    ## insert correlation
                    if np.abs(correlation) >= self.min_abs_correlation:
                        point_index_min, point_index_max = (min(point_index_i, point_index_j), max(point_index_i, point_index_j))
                        assert point_index_min < point_index_max and point_index_min in (point_index_i, point_index_j) and point_index_max in (point_index_i, point_index_j)
                        correlation_matrix[point_index_max, point_index_min] = correlation

        ## convert to wanted format
        if self.matrix_format != 'lil':
            logger.debug('Converting correlation matrix to format {}.'.format(self.matrix_format))
            if self.matrix_format == 'csc':
                correlation_matrix = correlation_matrix.asformat('csr')
                logger.debug('Correlation matrix converted to format csr.')
            correlation_matrix = correlation_matrix.asformat(self.matrix_format).astype(self.dtype)
            logger.debug('Correlation matrix converted to format {}.'.format(self.matrix_format))

        ## return
        logger.debug('Calculated differend boxes correlation lower triangle matrix with {} entries for minimal absolute correlation {} in matrix format {} with dtype {}.'.format(correlation_matrix.nnz, self.min_abs_correlation, self.matrix_format, self.dtype))
        return correlation_matrix


    ## correlation matrix

    @property
    def correlation_matrix(self):
        logger.debug('Calculating correlation matrix for minimal absolute correlation {} and maximal absolute correlation {} in matrix format {} with dtype {}.'.format(self.min_abs_correlation, self.max_abs_correlation, self.matrix_format, self.dtype))
        
        ## add same box and different boxes correlations lower triangle
        correlation_lower_triangle_matrix = self.different_boxes_correlation_lower_triangle_matrix + self.same_box_correlation_matrix_lower_triangle
        assert np.all(np.isclose(correlation_lower_triangle_matrix.diagonal(), 0))  

        ## apply max abs correlation
        mask = np.abs(correlation_lower_triangle_matrix.data) > self.max_abs_correlation
        correlation_lower_triangle_matrix.data[mask] = np.sign(correlation_lower_triangle_matrix.data[mask]) * self.max_abs_correlation
        
        ## add lower and upper triangle
        correlation_matrix = correlation_lower_triangle_matrix + correlation_lower_triangle_matrix.T
        
        ## add diagonal ones
        diagonal = scipy.sparse.eye(correlation_matrix.shape[0])
        correlation_matrix = correlation_matrix + diagonal
        
        ## return
        return correlation_matrix.asformat(self.matrix_format).astype(self.dtype)




class SampleCorrelationMatrixCache(SampleCorrelationMatrix):
    
    ## cachable values

    @util.cache.file_based.decorator()
    def map_indices_to_point_index_dict(self, discard_year=False):
        return super().map_indices_to_point_index_dict(discard_year=discard_year)
    
    @property
    @util.cache.file_based.decorator()
    def concentrations_same_points_except_year_dict(self):
        return super().concentrations_same_points_except_year_dict

    @property
    @util.cache.file_based.decorator()
    def sample_covariance_dict(self, value_type):
        return super().sample_covariance_dict
    
    @property
    @util.cache.file_based.decorator()
    def same_box_correlation_matrix_lower_triangle(self):
        return super().same_box_correlation_matrix_lower_triangle
    
    @property
    @util.cache.file_based.decorator()
    def different_boxes_quantity_lower_triangle_matrix(self):
        return super().different_boxes_quantity_lower_triangle_matrix
    
    @property
    @util.cache.file_based.decorator()
    def different_boxes_correlation_lower_triangle_matrix(self):
        return super().different_boxes_correlation_lower_triangle_matrix
    
    @property
    @util.cache.file_based.decorator()
    def correlation_matrix(self):
        return super().correlation_matrix
    
    
    ## cache files
    
    def map_indices_to_point_index_dict_cache_file(self, discard_year=False):
        return measurements.universal.constants.MAP_INDEX_TO_POINT_INDEX_DICT_FILE.format(tracer=self.measurements.tracer, data_set=self.measurements.data_set_name, sample_lsm=self.sample_lsm, discard_year=discard_year)
    
    def concentrations_same_points_except_year_dict_cache_file(self):
        return measurements.universal.constants.CONCENTRATIONS_SAME_POINTS_EXCEPT_YEAR_DICT_FILE.format(tracer=self.measurements.tracer, data_set=self.measurements.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements)
    
    def sample_covariance_dict_cache_file(self):
        return measurements.universal.constants.SAMPLE_COVARIANCE_DICT_FILE.format(tracer=self.measurements.tracer, data_set=self.measurements.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements, max_year_diff=self.max_year_diff)
    
    def same_box_correlation_matrix_lower_triangle_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements=self.measurements.min_measurements_standard_deviation, min_standard_deviation=self.measurements.min_standard_deviation, fill_strategy=self.measurements._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.SAMPLE_CORRELATION_MATRIX_SAME_BOX_LOWER_TRIANGLE_MATRIX_FILE.format(tracer=self.measurements.tracer, data_set=self.measurements.data_set_name, sample_lsm=self.sample_lsm, min_abs_correlation=self.min_abs_correlation, standard_deviation_description=standard_deviation_description, dtype=self.dtype, matrix_format=self.matrix_format)    
    
    def different_boxes_quantity_lower_triangle_matrix_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements=self.measurements.min_measurements_standard_deviation, min_standard_deviation=self.measurements.min_standard_deviation, fill_strategy=self.measurements._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.SAMPLE_QUANTITY_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE.format(tracer=self.measurements.tracer, data_set=self.measurements.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements, max_year_diff=self.max_year_diff, min_abs_correlation=self.min_abs_correlation, standard_deviation_description=standard_deviation_description, dtype=self.dtype, matrix_format=self.matrix_format)
    
    def different_boxes_correlation_lower_triangle_matrix_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements=self.measurements.min_measurements_standard_deviation, min_standard_deviation=self.measurements.min_standard_deviation, fill_strategy=self.measurements._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.SAMPLE_CORRELATION_MATRIX_DIFFERENT_BOXES_LOWER_TRIANGLE_MATRIX_FILE.format(tracer=self.measurements.tracer, data_set=self.measurements.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements, max_year_diff=self.max_year_diff, min_abs_correlation=self.min_abs_correlation, standard_deviation_description=standard_deviation_description, dtype=self.dtype, matrix_format=self.matrix_format)
    
    def correlation_matrix_cache_file(self):
        standard_deviation_description = measurements.universal.constants.DEVIATION_DESCRIPTION.format(min_measurements=self.measurements.min_measurements_standard_deviation, min_standard_deviation=self.measurements.min_standard_deviation, fill_strategy=self.measurements._fill_strategy_str('standard_deviations'))
        return measurements.universal.constants.SAMPLE_CORRELATION_MATRIX_FILE.format(tracer=self.measurements.tracer, data_set=self.measurements.data_set_name, sample_lsm=self.sample_lsm, min_measurements_correlation=self.min_measurements, max_year_diff=self.max_year_diff, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, standard_deviation_description=standard_deviation_description, dtype=self.dtype, matrix_format=self.matrix_format)
