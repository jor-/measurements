import numpy as np
import scipy.sparse

import measurements.all.pw.values
import measurements.dop.pw.correlation
import measurements.po4.wod.correlation.model
import measurements.po4.wod.correlation.estimation
import measurements.util.correlation
import measurements.util.data

import util.math.sparse.create
import util.cache
import util.logging
logger = util.logging.logger

from measurements.constants import CORRELATION_MIN_ABS_VALUE, CORRELATION_MAX_ABS_VALUE


class CorrelationModel(measurements.util.correlation.Model):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.args = args
        self.kargs = kargs


    @property
    def m(self):
        return 2


    @property
    def n(self):
        try:
            self._n
        except AttributeError:
            self._n = sum(map(len, self.points))

        return self._n


    @property
    def points(self):
        try:
            self._points
        except AttributeError:
            self._points = measurements.all.pw.values.points()
            assert len(self._points) == self.m

        return self._points


    @property
    def models(self):
        try:
            self._models
        except AttributeError:
            self._models = (measurements.dop.pw.correlation.CorrelationModel(*self.args, **self.kargs), measurements.po4.wod.correlation.model.CorrelationModel(*self.args, **self.kargs))
            assert len(self._models) == self.m

        return self._models


    @property
    def number_of_sample_values(self):
        return sum(map(len, self.models))



    def tracer_and_value_index(self, index):
        tracer_index = 0
        value_index = index
        while len(self.points[tracer_index]) <= value_index:
            value_index -= len(self.points[tracer_index])
            tracer_index += 1
        return (tracer_index, value_index)


    def tracer_and_value_indices(self, indices):
        tracer_and_value_indices = []
        for index in indices:
            tracer_and_value_indices.append(self.tracer_and_value_index(index))
        return tracer_and_value_indices


    # def sample_value(self, key):
    #     assert len(keys) == 2
    #
    #     ## if same point return 1
    #     if np.all(keys[0] == keys[1]):
    #         quantity = self.same_box_quantity
    #         correlation = 1
    #
    #     ## categorize
    #     else:
    #         keys = [list(keys[0]), list(keys[1])]
    #
    #         for i in range(len(keys)):
    #             keys[i] = measurements.util.data.Measurements.categorize_index(keys[i], self.SAMPLE_LSM.separation_values, discard_year=False)
    #             keys[i] = self.sample_lsm.coordinate_to_map_index(*keys[i], discard_year=False)
    #             keys[i] = np.array(np.round(keys[i]), dtype=np.int32)
    #
    #         ## if same point return same box correlation
    #         if np.all(keys[0] == keys[1]):
    #             quantity = self.same_box_quantity
    #             correlation = self.same_box_correlation
    #
    #         ## otherwise return 0
    #         else:
    #             quantity = 0
    #             correlation = self.no_data_correlation
    #
    #     return (quantity, correlation)

    def value(self, indices):
        assert len(indices) == 2

        ## correlation 1 for same indices
        if indices[0] == indices[1]:
            quantity = self.same_box_quantity
            correlation = 1
            return (quantity, correlation)[self.return_index]
        else:
            converted_indices = self.tracer_and_value_indices(indices)

            ## same tracer
            if converted_indices[0][0] == converted_indices[1][0]:
                model = self.models[converted_indices[0][0]]
                keys = (self.points[converted_indices[0][0]][converted_indices[0][1]], self.points[converted_indices[1][0]][converted_indices[1][1]])
                # correlation = model[keys]
                return model[keys]
            ## different tracers
            else:
                quantity = 0
                correlation = 0
                return (quantity, correlation)[self.return_index]

        # return correlation


    def effective_max_year_diff(self, tracer_index):
        return self.models[tracer_index].effective_max_year_diff


# class CorrelationModel():
#
#     def __init__(self, min_values=5, max_year_diff=1, min_abs_correlation=CORRELATION_MIN_ABS_VALUE, max_abs_correlation=CORRELATION_MAX_ABS_VALUE):
#         self.min_values = min_values
#         self.max_year_diff = max_year_diff
#         self.min_abs_correlation = min_abs_correlation
#         self.max_abs_correlation = max_abs_correlation
#
#
#     @property
#     def m(self):
#         return 2
#
#
#     @property
#     def points(self):
#         try:
#             self._points
#         except AttributeError:
#             self._points = measurements.all.pw.values.points()
#             assert len(self._points) == self.m
#
#         return self._points
#
#
#     @property
#     def models(self):
#         try:
#             self._models
#         except AttributeError:
#             self._models = (measurements.dop.pw.correlation.CorrelationModel(min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation), measurements.po4.wod.correlation.model.CorrelationModel(min_values=self.min_values, max_year_diff=self.max_year_diff, min_abs_correlation=self.min_abs_correlation, max_abs_correlation=self.max_abs_correlation))
#             assert len(self._models) == self.m
#
#         return self._models
#
#
#     @property
#     def n(self):
#         try:
#             self._n
#         except AttributeError:
#             self._n = sum(map(len, self.points))
#
#         return self._n
#
#
#     def tracer_and_value_index(self, index):
#         tracer_index = 0
#         value_index = index
#         while len(self.points[tracer_index]) <= value_index:
#             value_index -= len(self.points[tracer_index])
#             tracer_index += 1
#         return (tracer_index, value_index)
#
#
#     def tracer_and_value_indices(self, indices):
#         tracer_and_value_indices = []
#         for index in indices:
#             tracer_and_value_indices.append(self.tracer_and_value_index(index))
#         return tracer_and_value_indices
#
#
#     def value(self, indices):
#         assert len(indices) == 2
#
#         ## correlation 1 for same indices
#         if indices[0] == indices[1]:
#             correlation = 1
#         else:
#             converted_indices = self.tracer_and_value_indices(indices)
#
#             ## same tracer
#             if converted_indices[0][0] == converted_indices[1][0]:
#                 model = self.models[converted_indices[0][0]]
#                 keys = (self.points[converted_indices[0][0]][converted_indices[0][1]], self.points[converted_indices[1][0]][converted_indices[1][1]])
#                 correlation = model[keys]
#                 # correlation = np.sign(correlation) * min(np.abs(correlation), self.max_abs_correlation)
#             ## different tracers
#             else:
#                 correlation = 0
#
#         return correlation
#
#
#     def __getitem__(self, key):
#         return self.value(key)
#
#
#     @property
#     def effective_max_year_diff(self):
#         return self.models[1].effective_max_year_diff



# class CorrelationMatrixOld():
#
#     def __init__(self, *args, max_processes=None, step_size=10**3, **kargs):
#         kargs['return_type'] = measurements.util.correlation.RETURN_QUANTITY_AND_CORRELATION
#         self.model = CorrelationModel(*args, **kargs)
#         self.max_processes = max_processes
#         self.step_size = step_size
#
#
#     @property
#     def n(self):
#         return self.model.n
#
#     @property
#     def shape(self):
#         return (self.n, self.n)
#
#     def dtype(self, matrix_type):
#         if matrix_type in ('coo', 'csr') :
#             return np.float32
#         else:
#             return np.float16
#
#
#     ## fill methods
#
#     def fill_matrix_partial(self, insert_function, *indices):
#         logger.debug('Preparing values for row/colum indices {}.'.format(indices))
#
#         ## unpack indices if passed as one argument
#         try:
#             len(indices)
#             assert len(indices) == 1
#             indices = indices[0]
#         except TypeError:
#             pass
#
#         ## insert value for each passed index
#         for i in indices:
#             # insert_function(i, i, 1)
#             (tracer_i, value_i) = self.model.tracer_and_value_index(i)
#             max_year_diff_i = self.model.effective_max_year_diff(tracer_i)
#             t_i = self.model.points[tracer_i][value_i][0]
#
#             j = i+1
#             stop = False
#             while j < self.n and not stop:
#                 (tracer_j, value_j) = self.model.tracer_and_value_index(j)
#                 t_j = self.model.points[tracer_j][value_j][0]
#
#                 if tracer_i == tracer_j and np.abs(t_i - t_j) < max_year_diff_i:
#                     values = self.model[(i,j)]
#                     assert len(values) in (1,2)
#
#                     if values[0] != 0:
#                         insert_function(j, i, values[0])
#                     if len(values) == 2 and values[1] != 0:
#                         insert_function(i, j, values[1])
#                     # correlation = self.model[(i,j)]
#                     # if correlation != 0:
#                     #     insert_function(i, j, correlation)
#                     #     insert_function(j, i, correlation)
#                     j = j+1
#                 else:
#                     stop = True
#
#         logger.debug('Values for row/column indices {} prepared.'.format(indices))
#
#
#     def fill_indexable_matrix(self, matrix):
#         logger.debug('Calculating {} of size {} with dtype {} and max_year_diff {}.'.format(matrix.__class__.__name__, matrix.shape, matrix.dtype, self.model.max_year_diff))
#
#         def insert_function(i, j, c):
#             matrix[i,j] = c
#
#         self.fill_matrix_partial(insert_function, range(self.n))
#         return matrix
#
#
#     def fill_coo_matrix_partial(self, *indices):
#         logger.debug('Preparing partial COO matrix for indices {}.'.format(indices))
#
#         ## init
#         row_indices = []
#         colum_indices = []
#         data = []
#
#         def insert_function(i, j, c):
#             row_indices.append(i)
#             colum_indices.append(j)
#             data.append(c)
#
#         ## fill partial
#         self.fill_matrix_partial(insert_function, *indices)
#
#         ## convert to COO matrix
#         data = np.asanyarray(data, dtype=self.dtype('coo'))
#         coo_matrix = scipy.sparse.coo_matrix((data, (row_indices, colum_indices)), shape=self.shape)
#
#         logger.debug('Partial COO matrix with {} elements for indices {} prepared.'.format(coo_matrix.getnnz(), indices))
#         assert coo_matrix.dtype == self.dtype('coo')
#         return coo_matrix
#
#
#     def fill_csr_matrix_partial(self, *indices):
#         logger.debug('Preparing partial CSR matrix for indices {}.'.format(indices))
#
#         coo_matrix = self.fill_coo_matrix_partial(*indices)
#         csr_matrix = coo_matrix.tocsr()
#
#         logger.debug('Partial CSR matrix with {} elements for indices {} prepared.'.format(csr_matrix.getnnz(), indices))
#         assert csr_matrix.dtype == self.dtype('csr')
#         return csr_matrix
#
#
#     ## matrices
#
#     def quantitiy_and_correlation_matrix_calculate(self):
#         ## calculate in parallel
#         step_size = self.step_size
#         indices = tuple([range(i*step_size, (i+1)*step_size) for i in range(int(self.n/step_size))] + [range(int(self.n/step_size)*step_size, self.n)])
#
#         ## load values in cache
#         self.model.points
#         self.model.models
#
#         ## calculate
#         matrix = util.math.sparse.create.csr_matrix(self.fill_csr_matrix_partial, self.shape, indices, dtype=self.dtype('csr'), number_of_processes=self.max_processes, chunksize=1)
#         return matrix
#
#
#     @property
#     def quantitiy_and_correlation_matrix(self):
#         from measurements.all.pw.constants import CORRELATION_DIR, MATRIX_FILENAME
#         cache = util.cache.HDD_ObjectCache(CORRELATION_DIR)
#         filename = MATRIX_FILENAME.format(min_values=self.model.min_values, max_year_diff=self.model.max_year_diff, type='quantitiy_and_correlation.csr')
#         return cache.get_value(filename, self.quantitiy_and_correlation_matrix_calculate)
#
#
#     def correlation_matrix_calculate(self):
#         quantitiy_and_correlation_matrix = self.quantitiy_and_correlation_matrix
#         correlation_matrix = scipy.sparse.tril(quantitiy_and_correlation_matrix, k=-1, format='csr')
#         correlation_matrix = correlation_matrix + scipy.sparse.eye(10) + correlation_matrix.transpose()
#         correlation_matrix = correlation_matrix.tocsr()
#         return correlation_matrix
#
#
#     @property
#     def correlation_matrix(self):
#         from measurements.all.pw.constants import CORRELATION_DIR, MATRIX_FILENAME
#         cache = util.cache.HDD_ObjectCache(CORRELATION_DIR)
#         filename = MATRIX_FILENAME.format(min_values=self.model.min_values, max_year_diff=self.model.max_year_diff, type='correlation.csr')
#         return cache.get_value(filename, self.correlation_matrix_calculate)
#
#
#     # ## DOK matrix
#     #
#     #
#     # def dok_matrix_calculate(self):
#     #     correlation_matrix = scipy.sparse.dok_matrix(self.shape, dtype=self.dtype('dok'))
#     #     return self.fill_indexable_matrix(correlation_matrix)
#     #
#     # @property
#     # def dok_matrix(self):
#     #     from measurements.all.pw.constants import CORRELATION_DIR, MATRIX_FILENAME
#     #     cache = util.cache.HDD_ObjectCache(CORRELATION_DIR)
#     #     filename = MATRIX_FILENAME.format(min_values=self.model.min_values, max_year_diff=self.model.max_year_diff, type='dok')
#     #     return cache.get_value(filename, self.dok_matrix_calculate)
#     #
#     #
#     #
#     # ## CSR matrix
#     #
#     # def csr_matrix_calculate(self, step_size=10**3):
#     #     ## calculate in parallel
#     #     indices = tuple([range(i*step_size, (i+1)*step_size) for i in range(int(self.n/step_size))] + [range(int(self.n/step_size)*step_size, self.n)])
#     #
#     #     ## load values in cache
#     #     self.model.points
#     #     self.model.models
#     #
#     #     ## calculate
#     #     csr_matrix = util.math.sparse.create.csr_matrix(self.fill_csr_matrix_partial, self.shape, indices, dtype=self.dtype('csr'), number_of_processes=self.max_processes, chunksize=1)
#     #     return csr_matrix
#     #
#     # @property
#     # def csr_matrix(self):
#     #     from measurements.all.pw.constants import CORRELATION_DIR, MATRIX_FILENAME
#     #     cache = util.cache.HDD_ObjectCache(CORRELATION_DIR)
#     #     filename = MATRIX_FILENAME.format(min_values=self.model.min_values, max_year_diff=self.model.max_year_diff, type='csr')
#     #     return cache.get_value(filename, self.csr_matrix_calculate)
#     #
#     #
#     # @property
#     # def csc_matrix(self):
#     #     from measurements.all.pw.constants import CORRELATION_DIR, MATRIX_FILENAME
#     #     cache = util.cache.HDD_ObjectCache(CORRELATION_DIR)
#     #     filename = MATRIX_FILENAME.format(min_values=self.model.min_values, max_year_diff=self.model.max_year_diff, type='csc')
#     #     return cache.get_value(filename, lambda :self.csr_matrix.tocsc())
#
#
#     ## decompositions
#
#     def cholesky_L_calculate(self):
#         raise NotImplementedError()
#
#     @property
#     def cholesky_L(self):
#         from measurements.all.pw.constants import CORRELATION_DIR, MATRIX_FILENAME
#         cache = util.cache.HDD_ObjectCache(CORRELATION_DIR)
#         # filename = CHOLESKY_L_MATRIX_FILENAME.format(self.model.min_values, self.model.max_year_diff)
#         filename = MATRIX_FILENAME.format(min_values=self.model.min_values, max_year_diff=self.model.max_year_diff, type='positive_definite.L.csc')
#         return cache.get_value(filename, self.cholesky_L_calculate)
#
#
#     def cholesky_P_calculate(self):
#         raise NotImplementedError()
#
#     @property
#     def cholesky_P(self):
#         from measurements.all.pw.constants import CORRELATION_DIR, MATRIX_FILENAME
#         cache = util.cache.HDD_ObjectCache(CORRELATION_DIR)
#         # filename = CHOLESKY_P_MATRIX_FILENAME.format(self.model.min_values, self.model.max_year_diff)
#         filename = MATRIX_FILENAME.format(min_values=self.model.min_values, max_year_diff=self.model.max_year_diff, type='positive_definite.P.csc')
#         return cache.get_value(filename, self.cholesky_P_calculate)






class CorrelationMatrix(measurements.util.correlation.Model):

    def __init__(self, *args, positive_definite_approximation_reorder_after_each_step=True, **kargs):
        super().__init__(*args, **kargs)
        self.args = args
        self.kargs = kargs
        self.reordering = positive_definite_approximation_reorder_after_each_step

        self.memory_cache = util.cache.MemoryCache()

        from measurements.dop.pw.constants import CORRELATION_DIR as DOP_CORRELATION_DIR
        from measurements.po4.wod.correlation.constants import VALUE_DIR as PO4_CORRELATION_DIR
        self.point_dict_caches = tuple((util.cache.HDD_ObjectWithSaveCache(dir, measurements.util.data.Measurements.load, use_memory_cache=True) for dir in (DOP_CORRELATION_DIR, PO4_CORRELATION_DIR)))

        from measurements.all.pw.constants import CORRELATION_DIR
        self.object_cache = util.cache.HDD_ObjectCache(CORRELATION_DIR, use_memory_cache=True)
        self.npy_cache = util.cache.HDD_NPY_Cache(CORRELATION_DIR, use_memory_cache=True)



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
    def dtype(self):
        return np.float32

    @property
    def format(self):
        return 'csc'

    @property
    def ordering_method(self):
        return 'default'

    # @property
    # def reordering(self):
    #     return True


    ##  point index dicts

    def point_index_dict_calculate(self, i, discard_year=False):
        logger.debug('Calculating point index dict for tracer {} with discard year {}.'.format(i, discard_year))

        points_i = self.points[i]
        point_index_dict = measurements.util.data.Measurements()
        l = len(points_i)

        for j in range(l):
            key = points_i[j]
            transformed_key = measurements.po4.wod.correlation.estimation.transform_key(key, discard_year=discard_year)
            point_index_dict.append_value(transformed_key, j)

        return point_index_dict


    def point_index_dicts_calculate(self, discard_year=False):
        from measurements.all.pw.constants import POINT_INDEX_FILENAME
        if discard_year:
            filename = POINT_INDEX_FILENAME.format('with_year')
        else:
            filename = POINT_INDEX_FILENAME.format('year_discarded')

        return tuple(self.point_dict_caches[i][(filename, lambda : self.point_index_dict_calculate(i, discard_year=discard_year))] for i in range(self.m))

    @property
    def point_index_dicts(self):
        return self.memory_cache[('point_index_dicts', lambda : self.point_index_dicts_calculate(discard_year=False))]

    @property
    def point_index_year_discarded_dicts(self):
        return self.memory_cache[('point_index_year_discarded_dicts', lambda : self.point_index_dicts_calculate(discard_year=True))]


    ## same box
    def same_box_correlation_matrix_lower_triangle_calculate(self, format='csc', dtype=np.float32):
        logger.debug('Calculating same box correlation maxtrix with format {}.'.format(format))

        matrix = util.math.sparse.create.InsertableMatrix(self.shape, dtype=dtype)

        for k in range(self.m):
            index_offset = self.index_offsets[k]
            point_index_dict = self.point_index_dicts[k]

            for key, value_list in point_index_dict.iterator_keys_and_value_lists():
                n = len(value_list)
                for i in range(n):
                    point_index_i = value_list[i] + index_offset
                    for j in range(i+1, n):
                        point_index_j = value_list[j] + index_offset
                        assert point_index_i < point_index_j
                        matrix.insert(point_index_j, point_index_i, self.same_box_correlation)

        return matrix.asformat(format)

    @property
    def same_box_correlation_matrix_lower_triangle(self):
        from measurements.all.pw.constants import CORRELATION_MATRIX_SAME_BOX_FILENAME
        filename = CORRELATION_MATRIX_SAME_BOX_FILENAME.format(type=self.format)
        lower_triangle =  self.object_cache[(filename, lambda : self.same_box_correlation_matrix_lower_triangle_calculate(format=self.format, dtype=self.dtype))]
        return lower_triangle

    @property
    def same_box_correlation_matrix(self):
        lower_triangle =  self.same_box_correlation_matrix_lower_triangle
        matrix = lower_triangle + lower_triangle.transpose()
        return matrix


    ## sample quantity and correlation maxtrix

    def sample_value_dicts_calculate(self):
        return (measurements.dop.pw.correlation.sample_values_transformed('correlation', self.min_values, self.max_year_diff), measurements.po4.wod.correlation.estimation.sample_values_transformed('correlation', self.min_values, self.max_year_diff))

    @property
    def sample_value_dicts(self):
        return self.memory_cache[('sample_value_dicts', lambda : self.sample_value_dicts_calculate())]


    def sample_quantity_and_correlation_matrix_calculate(self, format='csc', dtype=np.float32):
        logger.debug('Calculating sample correlation matrix with format {}.'.format(format))

        matrix = util.math.sparse.create.InsertableMatrix(self.shape, dtype=dtype)

        ## sample value

        for k in range(self.m):
            index_offset = self.index_offsets[k]
            points = self.points[k]
            point_index_dict = self.point_index_dicts[k]
            point_index_year_discarded_dict = self.point_index_year_discarded_dicts[k]
            sample_value_dict = self.sample_value_dicts[k]

            for key, value_list in sample_value_dict.iterator_keys_and_value_lists():
                assert len(key) == 2
                assert len(value_list) == 1
                values = value_list[0]
                assert len(values) == 2
                quantity, correlation = values
                key_array = np.array(key)
                key_diff = key_array[1] - key_array[0]
                logger.debug('Calculating for key {}, quantity {} and correlation {}.'.format(key, quantity, correlation))

                for point_index_i in point_index_year_discarded_dict[key[0]]:
                    point_i = points[point_index_i]
                    point_i_transformed = measurements.po4.wod.correlation.estimation.transform_key(point_i, discard_year=False)
                    point_j_transformed = tuple(point_i_transformed + key_diff)
                    for point_index_j in point_index_dict[point_j_transformed]:
                        point_index_min, point_index_max = (min(point_index_i, point_index_j), max(point_index_i, point_index_j))
                        assert point_index_min < point_index_max and point_index_min in (point_index_i, point_index_j) and point_index_max in (point_index_i, point_index_j)
                        point_index_min = point_index_min + index_offset
                        point_index_max = point_index_max + index_offset
                        matrix.insert(point_index_max, point_index_min, correlation)
                        matrix.insert(point_index_min, point_index_max, quantity)

        return matrix.asformat(format)


    @property
    def sample_quantity_and_correlation_matrix(self):
        from measurements.all.pw.constants import SAMPLE_CORRELATION_AND_QUANTITY_MATRIX_FILENAME

        # filename = SAMPLE_CORRELATION_AND_QUANTITY_MATRIX_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, type='coo')
        # quantity_and_correlation_matrix = self.object_cache[(filename, lambda : self.sample_quantity_and_correlation_matrix_calculate('coo'))]
        #
        # filename = SAMPLE_CORRELATION_AND_QUANTITY_MATRIX_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, type=self.format)
        # quantity_and_correlation_matrix = self.object_cache[(filename, lambda : quantity_and_correlation_matrix.asformat(self.format).astype(self.dtype))]

        filename = SAMPLE_CORRELATION_AND_QUANTITY_MATRIX_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, type=self.format)
        quantity_and_correlation_matrix = self.object_cache[(filename, lambda : self.sample_quantity_and_correlation_matrix_calculate(format=self.format, dtype=self.dtype))]
        return quantity_and_correlation_matrix


    @property
    def sample_correlation_matrix_calculate(self, format='csc', dtype=np.float32):
        format = 'csc'
        quantity_and_correlation_matrix = self.sample_quantity_and_correlation_matrix
        correlation_matrix = scipy.sparse.tril(quantity_and_correlation_matrix, k=-1).asformat(format)
        correlation_matrix = correlation_matrix + scipy.sparse.eye(self.n) + correlation_matrix.transpose()
        return correlation_matrix.astype(dtype)

    # @property
    def sample_correlation_matrix(self):
        return self.sample_correlation_matrix_calculate(format=self.format, dtype=self.dtype)


    @property
    def sample_quantity_matrix_calculate(self, format='csc', dtype=np.int32):
        quantity_and_correlation_matrix = self.sample_quantity_and_correlation_matrix
        quantity_matrix = scipy.sparse.triu(quantity_and_correlation_matrix, k=1).asformat(format)
        quantity_matrix.data = np.round(quantity_matrix.data)
        quantity_matrix = quantity_matrix + quantity_matrix.transpose()
        return quantity_matrix.astype(dtype)

    # @property
    def sample_quantity_matrix(self):
        return self.sample_quantity_matrix_calculate(format=self.format)


    ## correlation matrix

    def correlation_matrix_calculate(self, format='csc', dtype=np.float32):
        quantity_and_correlation_matrix = self.sample_quantity_and_correlation_matrix
        correlation_matrix_lower_triangle = scipy.sparse.tril(quantity_and_correlation_matrix).asformat(format)

        mask = np.abs(correlation_matrix_lower_triangle.data) < self.min_abs_correlation
        correlation_matrix_lower_triangle.data[mask] = 0
        correlation_matrix_lower_triangle.eliminate_zeros()

        mask = np.abs(correlation_matrix_lower_triangle.data) > self.max_abs_correlation
        correlation_matrix_lower_triangle.data[mask] = np.sign(correlation_matrix_lower_triangle.data[mask]) * self.max_abs_correlation

        correlation_matrix_lower_triangle = correlation_matrix_lower_triangle + self.same_box_correlation_matrix_lower_triangle
        correlation_matrix = correlation_matrix_lower_triangle + scipy.sparse.eye(self.n) + correlation_matrix_lower_triangle.transpose()

        return correlation_matrix.asformat(format).astype(dtype)

    # @property
    def correlation_matrix(self, use_memory_cache=True):
        from measurements.all.pw.constants import CORRELATION_MATRIX_FILENAME
        filename = CORRELATION_MATRIX_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, type=self.format)
        # correlation_matrix = self.object_cache[(filename, lambda : self.correlation_matrix_calculate(format=self.format, dtype=self.dtype), use_memory_cache=use_memory_cache)]
        correlation_matrix = self.object_cache.get_value(filename, lambda : self.correlation_matrix_calculate(format=self.format, dtype=self.dtype), use_memory_cache=use_memory_cache)

        return correlation_matrix


    ## positive definite correlation matrix

    def correlation_matrix_positive_definite_and_reduction_factors_calculate(self, ordering_method='default', reordering=True, format='csc', dtype=np.float32):
        import util.math.sparse.decompose.with_cholmod
        from measurements.all.pw.constants import CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME
        reduction_factors_file = self.npy_cache.get_file(CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, ordering_method=ordering_method, reordering=reordering))
        # correlation_matrix = self.correlation_matrix
        correlation_matrix, reduction_factors = util.math.sparse.decompose.with_cholmod.approximate_positive_definite(self.correlation_matrix(use_memory_cache=False), min_abs_value=self.min_abs_correlation, ordering_method=ordering_method, reorder_after_each_step=reordering, use_long=True, reduction_factors_file=reduction_factors_file)
        return correlation_matrix.asformat(format).astype(dtype), reduction_factors.astype(dtype)


    def correlation_matrix_positive_definite_calculate(self, ordering_method='default', reordering=True, format='csc', dtype=np.float32):
        correlation_matrix, reduction_factors = self.memory_cache[('reduction_factors_and_correlation_matrix_positive_definite', lambda : self.correlation_matrix_positive_definite_and_reduction_factors_calculate(ordering_method=ordering_method, reordering=reordering, format=format, dtype=dtype))]
        return correlation_matrix


    def correlation_matrix_positive_definite_reduction_factors_calculate(self, ordering_method='default', reordering=True, format='csc', dtype=np.float32):
        correlation_matrix, reduction_factors = self.memory_cache[('reduction_factors_and_correlation_matrix_positive_definite', lambda : self.correlation_matrix_positive_definite_and_reduction_factors_calculate(ordering_method=ordering_method, reordering=reordering, format=self.format))]
        return reduction_factors


    @property
    def correlation_matrix_positive_definite_and_reduction_factors(self):
        from measurements.all.pw.constants import CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME, CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME

        filename = CORRELATION_MATRIX_POSITIVE_DEFINITE_REDUCTION_FACTORS_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, ordering_method=self.ordering_method, reordering=self.reordering)
        reduction_factors = self.npy_cache[(filename, lambda : self.correlation_matrix_positive_definite_reduction_factors_calculate(ordering_method=self.ordering_method, reordering=self.reordering, format=self.format, dtype=self.dtype))]

        filename = CORRELATION_MATRIX_POSITIVE_DEFINITE_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, ordering_method=self.ordering_method, reordering=self.reordering, type=self.format)
        correlation_matrix = self.object_cache[(filename, lambda : self.correlation_matrix_positive_definite_calculate(ordering_method=self.ordering_method, reordering=self.reordering, format=self.format, dtype=self.dtype))]
        return correlation_matrix, reduction_factors

    @property
    def correlation_matrix_positive_definite(self):
        return self.correlation_matrix_positive_definite_and_reduction_factors[0]

    @property
    def correlation_matrix_positive_definite_reduction_factors(self):
        return self.correlation_matrix_positive_definite_and_reduction_factors[1]


    ## choleksy decomposition

    def correlation_matrix_cholesky_decomposition_calculate(self, ordering_method='default', format='csc', dtype=np.float32):
        import util.math.sparse.decompose.with_cholmod
        P, L = util.math.sparse.decompose.with_cholmod.cholesky(self.correlation_matrix_positive_definite, ordering_method=ordering_method, return_type=util.math.sparse.decompose.with_cholmod.RETURN_P_L, use_long=True)
        return P.asformat(format).astype(dtype), L.asformat(format).astype(dtype)

    @property
    def correlation_matrix_cholesky_decomposition(self):
        from measurements.all.pw.constants import CORRELATION_MATRIX_CHOLESKY_FACTORS_FILENAME
        filename = CORRELATION_MATRIX_CHOLESKY_FACTORS_FILENAME.format(min_values=self.min_values, max_year_diff=self.max_year_diff, ordering_method=self.ordering_method, reordering=self.reordering, type=self.format)
        P, L = self.object_cache[(filename, lambda : self.correlation_matrix_cholesky_decomposition_calculate(ordering_method=self.ordering_method, format=self.format, dtype=self.dtype))]
        return P, L



