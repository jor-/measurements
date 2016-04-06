import numpy as np
import scipy.sparse

import measurements.all.pw.values
import measurements.all.pw.constants as CONSTANTS
import measurements.dop.pw.correlation
import measurements.po4.wod.correlation.estimation
import measurements.util.correlation
import measurements.util.data

import util.math.sparse.create
import util.math.util
import util.cache
import util.logging
logger = util.logging.logger



class CorrelationMatrix:
    
    def __init__(self, min_measurements, max_year_diff=float('inf'), cholesky_ordering_method='default', positive_definite_approximation_reorder_after_each_step=True, positive_definite_approximation_min_diag_value=CONSTANTS.CORRELATION_MIN_DIAG_VALUE_POSITIVE_DEFINITE_APPROXIMATION, sample_lsm=CONSTANTS.SAMPLE_LSM, dtype=np.float32):
        
        assert positive_definite_approximation_min_diag_value > 0

        from measurements.constants import CORRELATION_MIN_ABS_VALUE, CORRELATION_MAX_ABS_VALUE

        self.min_measurements = min_measurements
        self.max_year_diff = max_year_diff
        self.ordering_method = cholesky_ordering_method
        self.min_abs_correlation = CORRELATION_MIN_ABS_VALUE
        self.max_abs_correlation = CORRELATION_MAX_ABS_VALUE
        self.reordering = positive_definite_approximation_reorder_after_each_step
        self.min_diag_value = positive_definite_approximation_min_diag_value

        self.memory_cache = util.cache.MemoryCacheDeactivatable()

        from measurements.dop.pw.constants import CORRELATION_DIR as DOP_CORRELATION_DIR
        from measurements.po4.wod.correlation.constants import VALUE_DIR as PO4_CORRELATION_DIR
        self.point_dict_caches = tuple((util.cache.HDD_ObjectWithSaveCache(dir, measurements.util.data.Measurements.load, use_memory_cache=True) for dir in (DOP_CORRELATION_DIR, PO4_CORRELATION_DIR)))

        from measurements.all.pw.constants import CORRELATION_DIR
        self.object_cache = util.cache.HDD_ObjectCache(CORRELATION_DIR, use_memory_cache=True)
        self.npy_cache = util.cache.HDD_NPY_Cache(CORRELATION_DIR, use_memory_cache=True)
        
        self.sample_lsm = sample_lsm
        self.dtype = dtype


    ## cache
    
    def memory_cache_switch(self, enabled):
        self.memory_cache.switch(enabled)
        self.object_cache.memory_cache_switch(enabled)
        self.npy_cache.memory_cache_switch(enabled)


    ## properties

    @property
    def points(self):
        return self.memory_cache[('points', measurements.all.pw.values.points)]

    @property
    def m(self):
        return self.memory_cache[('m', lambda : len(self.points))]

    @property
    def n(self):
        return self.memory_cache[('n', lambda : sum(map(len, self.points)))]

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def index_offsets(self):
        return self.memory_cache[('index_offsets', lambda : (0,) + tuple(map(len, self.points))[:-1])]

    @property
    def format(self):
        return 'csc'


    ##  point index dicts

    def point_index_dict_calculate(self, i, discard_year=False):
        logger.debug('Calculating point index dict for tracer {} with discard year {}.'.format(i, discard_year))

        points_i = self.points[i]
        point_index_dict = measurements.util.data.Measurements()
        l = len(points_i)

        for j in range(l):
            point_j = points_i[j]
            transformed_point_j = self.sample_lsm.coordinates_to_map_indices(point_j, discard_year=discard_year, float_indices=False)
            point_index_dict.append_value(transformed_point_j, j)

        return point_index_dict

    def point_index_dicts_calculate(self, discard_year=False):
        from measurements.all.pw.constants import POINT_INDEX_FILENAME
        if discard_year:
            filename = POINT_INDEX_FILENAME.format(year_type='with_year', sample_lsm=self.sample_lsm)
        else:
            filename = POINT_INDEX_FILENAME.format(year_type='year_discarded', sample_lsm=self.sample_lsm)

        return tuple(self.point_dict_caches[i][(filename, lambda : self.point_index_dict_calculate(i, discard_year=discard_year))] for i in range(self.m))

    @property
    def point_index_dicts(self):
        return self.memory_cache[('point_index_dicts', lambda : self.point_index_dicts_calculate(discard_year=False))]

    @property
    def point_index_year_discarded_dicts(self):
        return self.memory_cache[('point_index_year_discarded_dicts', lambda : self.point_index_dicts_calculate(discard_year=True))]



    ## different boxes covariance and quantity matrix

    def different_boxes_sample_covariance_dicts_calculate(self):
        return (measurements.dop.pw.correlation.different_boxes_sample_covariances_dict_transformed(self.min_measurements, self.max_year_diff), measurements.po4.wod.correlation.estimation.different_boxes_sample_covariances_dict_transformed(self.min_measurements, self.max_year_diff))

    @property
    def different_boxes_sample_covariance_dicts(self):
        return self.memory_cache[('different_boxes_sample_covariance_dicts', lambda : self.different_boxes_sample_covariance_dicts_calculate())]



    ## standard deviation

    @property
    def sample_deviations(self):
        deviations = np.concatenate(measurements.all.pw.values.deviation())
        logger.debug('Got sample deviations with length {}.'.format(len(deviations)))
        assert np.all(deviations > 0)
        return deviations

    @property
    def diag_sample_covariance_matrix(self):
        deviations = self.sample_deviations
        matrix = scipy.sparse.diags(deviations**2, offsets=0)
        logger.debug('Got diag sample covariance matrix with {} entries.'.format(matrix.nnz))
        return matrix

    @property
    def diag_inverse_sample_deviation_matrix(self):
        deviations = self.sample_deviations
        matrix = scipy.sparse.diags(deviations**(-1), offsets=0)
        logger.debug('Got diag inverse sample deviation matrix with {} entries.'.format(matrix.nnz))
        return matrix
    

    ## same box correlation
    
    def same_box_correlation_matrix_lower_triangle_calculate(self, min_abs_correlation=0, format='csc', dtype=np.float32):
        logger.debug('Calculating same box correlation matrix lower triangle with minimal absolute correlation {} in matrix format {} with dtype {}.'.format(min_abs_correlation, format, dtype))
        
        ## calculate
        same_box_correlation_matrix_lower_triangle = self.same_box_sample_covariance_matrix_lower_triangle
        inverse_deviation_matrix = self.diag_inverse_sample_deviation_matrix
        same_box_correlation_matrix_lower_triangle = inverse_deviation_matrix * same_box_correlation_matrix_lower_triangle * inverse_deviation_matrix

        ## apply min abs correlation
        mask = np.abs(same_box_correlation_matrix_lower_triangle.data) < min_abs_correlation
        same_box_correlation_matrix_lower_triangle.data[mask] = 0
        same_box_correlation_matrix_lower_triangle.eliminate_zeros()
        
        ## return
        return same_box_correlation_matrix_lower_triangle.asformat(format).astype(dtype)


    def same_box_correlation_matrix_lower_triangle(self, min_abs_correlation=None, format=None, dtype=None):
        ## if no value passed, use default value
        if min_abs_correlation is None:
            min_abs_correlation = self.min_abs_correlation
        if format is None:
            format = self.format
        if dtype is None:
            dtype = self.dtype
        
        ## calculate
        filename = CONSTANTS.SAME_BOX_CORRELATION_LOWER_TRIANGLE_MATRIX_FILENAME.format(min_abs_correlation=min_abs_correlation, sample_lsm=self.sample_lsm, matrix_type=format, dtype=dtype)
        lower_triangle = self.object_cache[(filename, lambda: self.same_box_correlation_matrix_lower_triangle_calculate(min_abs_correlation=min_abs_correlation, format=format, dtype=dtype))]
        
        ## return
        logger.debug('Got same box correlation matrix lower triangle with {} entries for minimal absolute correlation {} in matrix format {} with dtype {}.'.format(lower_triangle.nnz, min_abs_correlation, format, dtype))
        assert np.all(np.abs(lower_triangle.data) >= min_abs_correlation)
        return lower_triangle



    ## different boxes correlation and quantity matrix

    def different_boxes_quantity_lower_triangle_matrix_calculate(self, min_abs_correlation=0, format='csc'):
        logger.debug('Calculating different boxes quantity lower triangle matrix with minimal absolute correlation {} in matrix format {}.'.format(min_abs_correlation, format))

        ## get max quantity
        max_quantity = 0
        for k in range(self.m):
            sample_covariance_dict = self.different_boxes_sample_covariance_dicts[k]
            for key, value_list in sample_covariance_dict.iterator_keys_and_value_lists():
                assert len(key) == 2
                assert len(value_list) == 1
                values = value_list[0]
                assert len(values) == 2
                quantity, covariance = values
                if quantity > max_quantity:
                    max_quantity = quantity
        
        ## get matrix dtype
        dtype = util.math.util.min_int_dtype(max_quantity, unsigned=True)
        
        if format == 'lil':
            ## create matrix
            quantity_matrix = scipy.sparse.lil_matrix(self.shape, dtype=dtype)
            
            ## get deviation
            deviations = self.sample_deviations
            
            ## iterate over tracer
            for k in range(self.m):
                index_offset = self.index_offsets[k]
                points = self.points[k]
                point_index_dict = self.point_index_dicts[k]
                point_index_year_discarded_dict = self.point_index_year_discarded_dicts[k]
                sample_covariance_dict = self.different_boxes_sample_covariance_dicts[k]
            
                ## iterate over sample covariances
                for key, value_list in sample_covariance_dict.iterator_keys_and_value_lists():
                    assert len(key) == 2
                    assert len(value_list) == 1
                    values = value_list[0]
                    assert len(values) == 2
                    quantity, covariance = values
                    key_array = np.array(key)
                    key_diff = key_array[1] - key_array[0]
                    logger.debug('For different box correlation matrix entires with key {} inserting quantity {} (covariance {}).'.format(key, quantity, covariance))
                    
                    ## iterate over all index pairs with sample covariance
                    for point_index_i in point_index_year_discarded_dict[key[0]]:
                        point_i = points[point_index_i]
                        point_i_transformed = self.sample_lsm.coordinate_to_map_index(*point_i, discard_year=False, float_indices=False)
                        point_j_transformed = tuple(point_i_transformed + key_diff)
                        for point_index_j in point_index_dict[point_j_transformed]:
                            
                            ## calculate correlation
                            correlation = covariance / (deviations[point_index_i] * deviations[point_index_j])
                            
                            ## insert quantity
                            if np.abs(correlation) >= min_abs_correlation:
                                point_index_min, point_index_max = (min(point_index_i, point_index_j), max(point_index_i, point_index_j))
                                assert point_index_min < point_index_max and point_index_min in (point_index_i, point_index_j) and point_index_max in (point_index_i, point_index_j)
                                point_index_min = point_index_min + index_offset
                                point_index_max = point_index_max + index_offset
                                quantity_matrix[point_index_max, point_index_min] = quantity

        else:
            quantity_matrix = self.different_boxes_quantity_lower_triangle_matrix(min_abs_correlation=min_abs_correlation, format='lil')
            logger.debug('Converting matrix to format {} and dtype {}.'.format(format, dtype))
            if format == 'csc':
                quantity_matrix = self.different_boxes_quantity_lower_triangle_matrix(min_abs_correlation=min_abs_correlation, format='csr')
            quantity_matrix = quantity_matrix.asformat(format).astype(dtype)

        logger.debug('Calculated differend boxes quantity lower triangle matrices with each {} entries for minimal absolute correlation {}.'.format(correlation_matrix.nnz, min_abs_correlation))

        return quantity_matrix
    
    
    def different_boxes_quantity_lower_triangle_matrix(self, min_abs_correlation=None, format='lil'):
        if min_abs_correlation == None:
            min_abs_correlation = self.min_abs_correlation
        
        ## prepare file names
        filename = CONSTANTS.DIFFERENT_BOXES_QUANTITY_LOWER_TRIANGLE_MATRIX_FILENAME.format(min_abs_correlation=min_abs_correlation, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements, max_year_diff=self.max_year_diff, matrix_type=format)
        
        ## get value
        quantity_matrix = self.object_cache.get_value(filename, lambda: self.different_boxes_quantity_lower_triangle_matrix_calculate(min_abs_correlation=min_abs_correlation, format=format))
        
        ## return
        logger.debug('Got differend boxes quantity lower triangle matrix with {} entries for minimal absolute correlation {} in matrix format {}.'.format(quantity_matrix.nnz, min_abs_correlation, format))
        return quantity_matrix
    
    
    
    def different_boxes_correlation_lower_triangle_matrix_calculate(self, min_abs_correlation=0, format='csc', dtype=np.float32):
        logger.debug('Calculating different boxes correlation lower triangle matrix with minimal absolute correlation {} in matrix format {} with dtype {}.'.format(min_abs_correlation, format, dtype))
        
        if format == 'lil':
            ## create matrix
            correlation_matrix = scipy.sparse.lil_matrix(self.shape, dtype=dtype)
            
            ## get deviation
            deviations = self.sample_deviations
            
            ## iterate over tracer
            for k in range(self.m):
                index_offset = self.index_offsets[k]
                points = self.points[k]
                point_index_dict = self.point_index_dicts[k]
                point_index_year_discarded_dict = self.point_index_year_discarded_dicts[k]
                sample_covariance_dict = self.different_boxes_sample_covariance_dicts[k]
            
                ## iterate over sample covariances
                for key, value_list in sample_covariance_dict.iterator_keys_and_value_lists():
                    assert len(key) == 2
                    assert len(value_list) == 1
                    values = value_list[0]
                    assert len(values) == 2
                    quantity, covariance = values
                    key_array = np.array(key)
                    key_diff = key_array[1] - key_array[0]
                    logger.debug('For different box correlation matrix entires with key {} inserting covariance {} (quantity {}).'.format(key, covariance, quantity))
                    
                    ## iterate over all index pairs with sample covariance
                    for point_index_i in point_index_year_discarded_dict[key[0]]:
                        point_i = points[point_index_i]
                        point_i_transformed = self.sample_lsm.coordinate_to_map_index(*point_i, discard_year=False, float_indices=False)
                        point_j_transformed = tuple(point_i_transformed + key_diff)
                        for point_index_j in point_index_dict[point_j_transformed]:
                            
                            ## calculate correlation
                            correlation = covariance / (deviations[point_index_i] * deviations[point_index_j])
                            
                            ## insert correlation
                            if np.abs(correlation) >= min_abs_correlation:
                                point_index_min, point_index_max = (min(point_index_i, point_index_j), max(point_index_i, point_index_j))
                                assert point_index_min < point_index_max and point_index_min in (point_index_i, point_index_j) and point_index_max in (point_index_i, point_index_j)
                                point_index_min = point_index_min + index_offset
                                point_index_max = point_index_max + index_offset
                                correlation_matrix[point_index_max, point_index_min] = correlation

        else:
            correlation_matrix = self.different_boxes_correlation_lower_triangle_matrix(min_abs_correlation=min_abs_correlation, format='lil', dtype=dtype)
            logger.debug('Converting matrix to format {} and dtype {}.'.format(format, dtype))
            if format == 'csc':
                correlation_matrix = self.different_boxes_correlation_lower_triangle_matrix(min_abs_correlation=min_abs_correlation, format='csr', dtype=dtype)
            correlation_matrix = correlation_matrix.asformat(format).astype(dtype)

        logger.debug('Calculated differend boxes correlation lower triangle matrices with {} entries for minimal absolute correlation {} in matrix format {} with dtype {}.'.format(correlation_matrix.nnz, min_abs_correlation, format, dtype))
        # assert format not in ('csr', 'csc', 'coo') or np.all(np.abs(correlation_matrix.data) >= min_abs_correlation)
        return correlation_matrix


    def different_boxes_correlation_lower_triangle_matrix(self, min_abs_correlation=None, format=None, dtype=None):
        ## if no value passed, use default value
        if min_abs_correlation is None:
            min_abs_correlation = self.min_abs_correlation
        if format is None:
            format = self.format
        if dtype is None:
            dtype = self.dtype
        
        ## prepare file names
        filename = CONSTANTS.DIFFERENT_BOXES_CORRELATION_LOWER_TRIANGLE_MATRIX_FILENAME.format(min_abs_correlation=min_abs_correlation, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements, max_year_diff=self.max_year_diff, matrix_type=format, dtype=dtype)
        
        ## get value
        correlation_matrix = self.object_cache.get_value(filename, lambda: self.different_boxes_correlation_lower_triangle_matrix_calculate(min_abs_correlation=min_abs_correlation, format=format, dtype=dtype))
        
        ## return
        logger.debug('Got differend boxes correlation lower triangle matrix with {} entries for minimal absolute correlation {} in matrix format {} with dtype {}.'.format(correlation_matrix.nnz, min_abs_correlation, format, dtype))
        return correlation_matrix



    ## correlation matrix

    def correlation_matrix_calculate(self, min_abs_correlation=0, max_abs_correlation=1, format='csc', dtype=np.float32):
        logger.debug('Calculating correlation matrix for minimal absolute correlation {} and maximal absolute correlation {} in matrix format {} with dtype {}.'.format(min_abs_correlation, max_abs_correlation, format, dtype))
        
        ## add same box and different boxes correlations lower triangle
        same_box_correlation_matrix_lower_triangle = self.same_box_correlation_matrix_lower_triangle(min_abs_correlation=min_abs_correlation)
        different_boxes_correlation_lower_triangle_matrix = self.different_boxes_correlation_lower_triangle_matrix(min_abs_correlation=min_abs_correlation)
        # assert set(zip(*same_box_correlation_matrix.nonzero())).isdisjoint(set(zip(*different_boxes_correlation_matrix.nonzero())))
        correlation_lower_triangle_matrix = same_box_correlation_matrix_lower_triangle + different_boxes_correlation_lower_triangle_matrix
        assert np.all(np.isclose(correlation_lower_triangle_matrix.diagonal(), 0))  

        ## apply max abs correlation
        mask = np.abs(correlation_lower_triangle_matrix.data) > max_abs_correlation
        correlation_lower_triangle_matrix.data[mask] = np.sign(correlation_lower_triangle_matrix.data[mask]) * max_abs_correlation
        
        ## add lower and upper triangle
        correlation_matrix = correlation_lower_triangle_matrix + correlation_lower_triangle_matrix.T
        
        ## add diagonal ones
        diagonal = scipy.sparse.diags(np.ones(correlation_matrix.shape[0]), offsets=0)
        correlation_matrix = correlation_matrix + diagonal
        
        ## return
        assert np.all(np.isclose(correlation_matrix.diagonal(), 1))        
        assert np.isclose(np.abs(correlation_matrix.data).min(), min_abs_correlation) or np.abs(correlation_matrix.data).min() >= min_abs_correlation
        return correlation_matrix.asformat(format).astype(dtype)


    def correlation_matrix(self, min_abs_correlation=None, max_abs_correlation=None, use_memory_cache=True, format=None, dtype=None):
        ## if no value passed, use default value
        if min_abs_correlation is None:
            min_abs_correlation = self.min_abs_correlation
        if max_abs_correlation is None:
            max_abs_correlation = self.max_abs_correlation
        if format is None:
            format = self.format
        if dtype is None:
            dtype = self.dtype
        
        ## calculate
        filename = CONSTANTS.CORRELATION_MATRIX_FILENAME.format(min_abs_correlation=min_abs_correlation, max_abs_correlation=max_abs_correlation, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements, max_year_diff=self.max_year_diff, matrix_type=format, dtype=dtype)
        correlation_matrix = self.object_cache.get_value(filename, lambda: self.correlation_matrix_calculate(min_abs_correlation=min_abs_correlation, max_abs_correlation=max_abs_correlation, format=format, dtype=dtype), use_memory_cache=use_memory_cache)
        
        ## return
        logger.debug('Got correlation matrix with {} entries for minimal absolute correlation {} and maximal absolute correlation {}.'.format(correlation_matrix.nnz, min_abs_correlation, max_abs_correlation))
        assert np.all(np.abs(correlation_matrix.data) >= min_abs_correlation)
        assert np.all(np.abs(correlation_matrix.data) <= 1)
        return correlation_matrix



    ## positive definite correlation matrix

    def correlation_matrix_positive_definite_and_reduction_factors_calculate(self):
        import util.math.sparse.decompose.with_cholmod
        from measurements.all.pw.constants import CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME
        reduction_factors_file = self.npy_cache.get_file(CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME.format(min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements, max_year_diff=self.max_year_diff, ordering_method=self.ordering_method, reordering=self.reordering, min_diag_value=self.min_diag_value, dtype=self.dtype))
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlation_matrix(use_memory_cache=False), min_abs_value=self.min_abs_correlation, min_diag_value=self.min_diag_value, ordering_method=self.ordering_method, reorder_after_each_step=self.reordering, use_long=True, reduction_factors_file=reduction_factors_file)
        return correlation_matrix.asformat(self.format).astype(self.dtype), reduction_factors.astype(self.dtype)


    def correlation_matrix_positive_definite_calculate(self):
        correlation_matrix, reduction_factors = self.memory_cache[('reduction_factors_and_correlation_matrix_positive_definite', self.correlation_matrix_positive_definite_and_reduction_factors_calculate)]
        return correlation_matrix


    def correlation_matrix_positive_definite_reduction_factors_calculate(self):
        correlation_matrix, reduction_factors = self.memory_cache[('reduction_factors_and_correlation_matrix_positive_definite', self.correlation_matrix_positive_definite_and_reduction_factors_calculate)]
        return reduction_factors


    @property
    def correlation_matrix_positive_definite_and_reduction_factors(self):
        from measurements.all.pw.constants import CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME, CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME

        filename = CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME.format(min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements, max_year_diff=self.max_year_diff, ordering_method=self.ordering_method, reordering=self.reordering, min_diag_value=self.min_diag_value, dtype=self.dtype)
        reduction_factors = self.npy_cache[(filename, self.correlation_matrix_positive_definite_reduction_factors_calculate)]

        filename = CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME.format(min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements, max_year_diff=self.max_year_diff, ordering_method=self.ordering_method, reordering=self.reordering, min_diag_value=self.min_diag_value, matrix_type=self.format, dtype=self.dtype)
        correlation_matrix = self.object_cache[(filename, self.correlation_matrix_positive_definite_calculate)]

        return correlation_matrix, reduction_factors


    @property
    def correlation_matrix_positive_definite(self):
        return self.correlation_matrix_positive_definite_and_reduction_factors[0]

    @property
    def correlation_matrix_positive_definite_reduction_factors(self):
        return self.correlation_matrix_positive_definite_and_reduction_factors[1]



    ## choleksy decomposition

    def correlation_matrix_cholesky_decomposition_calculate(self):
        import util.math.sparse.decompose.with_cholmod
        P, L = util.math.sparse.decompose.with_cholmod.cholesky(self.correlation_matrix_positive_definite, ordering_method=self.ordering_method, return_type=util.math.sparse.decompose.with_cholmod.RETURN_P_L, use_long=True)
        return P.asformat(self.format).astype(self.dtype), L.asformat(self.format).astype(self.dtype)

    @property
    def correlation_matrix_cholesky_decomposition(self):
        from measurements.all.pw.constants import CORRELATION_MATRIX_CHOLESKY_FACTORS_FILENAME
        filename = CORRELATION_MATRIX_CHOLESKY_FACTORS_FILENAME.format(min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation, sample_lsm=self.sample_lsm, min_measurements=self.min_measurements, max_year_diff=self.max_year_diff, ordering_method=self.ordering_method, reordering=self.reordering, min_diag_value=self.min_diag_value, matrix_type=self.format, dtype=self.dtype)
        P, L = self.object_cache[(filename, self.correlation_matrix_cholesky_decomposition_calculate)]
        return P, L


