import numpy as np
import math
import bisect
import itertools
import logging


import util.io
import measurements.util.calculate

logger = logging.getLogger(__name__)



class Measurements_Unsorted():
    
    def __init__(self):
        self.measurements_dict = dict()
    
    
    def add_result(self, index, result):
        dictionary = self.measurements_dict
        measurements_dict_type = type(dictionary)
        
        n = len(index)
        for i in range(n-1):
            dictionary = dictionary.setdefault(index[i], measurements_dict_type())
        result_list = dictionary.setdefault(index[n-1], [])
        try:
            result_list.extend(result)
        except TypeError:
            result_list.append(result)
    
    def add_results(self, indices, results):
        results_len = len(results)
        logger.debug('Adding {} measurements.'.format(results_len))
        for i in range(results_len):
            self.add_result(indices[i], results[i])
    
    
    def save(self, file):
        logger.debug('Saving measurements at %s.', file)
        util.io.save_object(self.measurements_dict, file)
    
    
    def load(self, file):
        logger.debug('Loading measurements from %s.', file)
        self.measurements_dict = util.io.load_object(file)
    
    
    def get_value(self, index):
        value = self.measurements_dict
        
        for i in index:
            value = value[i]
        
        return value
        
    
    
    ## transform indices
    
    def transform_indices(self, transform_function):
        measurements_dict = self.measurements_dict
        measurements_dict_type = type(measurements_dict)
        measurements_dict_transformed = measurements_dict_type()
        self.measurements_dict = measurements_dict_transformed
        
        for (t, t_dict) in  measurements_dict.items():
            for (x, x_dict) in t_dict.items():
                for (y, y_dict) in x_dict.items():
                    for (z, results) in y_dict.items():
                        index = (t, x, y, z)
                        index_transformed = transform_function(index)
                        
                        self.add_result(index_transformed, results)
    
    
    
    
    def categorize_indices(self, separation_values, discard_year=False):
        def categorize_index(index, separation_values, discard_year=False):
            index = list(index)
            
            ## discard year
            if discard_year:
                index[0] = index[0] % 1
            
            ## iterate over dimensions
            for i in range(len(separation_values)):
                
                ## get separation value
                try:
                    separation_value = separation_values[i]
                except IndexError:
                    separation_value = None
                
                ## categorize dimension i
                if separation_value is not None:
                    # check if sequence
                    try:
                        is_sequence = len(separation_value) >= 2
                    except TypeError:
                        is_sequence = False
                    
                    # separate by value
                    if not is_sequence:
                        index[i] = (math.floor(index[i] / separation_value) + 0.5) * separation_value
                    # separate by sequence
                    else:
                        index_right = bisect.bisect_right(separation_value, index[i])
                        value_left = separation_value[index_right - 1]
                        try:
                            value_right = separation_value[index_right]
                        except IndexError:
                            raise ValueError('Index value %d exceeds range of separation values (right sight of separation values is %d).' % (index[i], value_left))
                        index[i] = (value_left + value_right) / 2.0
                    
#                     # wrap around
#                     if wrap_around_ranges is not None:
#                         try:
#                             wrap_around_range = wrap_around_ranges[i]
#                         except IndexError:
#                             wrap_around_range = None
#                     else:
#                         wrap_around_range = None
#                     
#                     if wrap_around_range is not None:
#                         index[i] = measurements.util.calculate.wrap_around_index(index[i], wrap_around_range)
            
            index = tuple(index)
            return index
        
        
        if discard_year:
            logger.debug('Indices categorized by separation values %s and discard year.' % str(separation_values))
        else:
            logger.debug('Indices categorized by separation values %s.' % str(separation_values))
        
#         transform_function = lambda index: categorize_index(index, separation_values, wrap_around_ranges=wrap_around_ranges, discard_year=discard_year)
        transform_function = lambda index: categorize_index(index, separation_values, discard_year=discard_year)
        
        self.transform_indices(transform_function)
    
    
    def transform_indices_to_boxes(self, x_dim, y_dim, z_values):
        def transform_index_to_boxes(index, x_dim, y_dim, z_values):
            def transform_space_index(index, range, new_len):
                index = np.floor(((index - range[0]) / (range[1] - range[0])) * new_len)
                if index == new_len:
                    index -= 1
                return index
            
            index = list(index)
            
            if index[1] < 0:
                index[1] += 360
            index[1] = transform_space_index(index[1], (0, 360), x_dim)
            index[2] = transform_space_index(index[2], (-90, 90), y_dim)
            index[3] = bisect.bisect_right(z_values, index[3]) - 1
            
            index = tuple(index)
            
            return index
        
        
        logger.debug('Transform indices to boxes with x_dim %d, y_dim %d and z_values %s.' % (x_dim, y_dim, str(z_values)))
        
        transform_function = lambda index: transform_index_to_boxes(index, x_dim=x_dim, y_dim=y_dim, z_values=z_values)
        
        self.transform_indices(transform_function)
    
    
    def discard_year(self):
        logger.debug('Discarding year.')
        
        def transform_function(index):
            index_list = list(index)
            index_list[0] = index[0] % 1
            index = tuple(index_list)
            return index
            
        self.transform_indices(transform_function)
    
    
    def discard_time(self):
        logger.debug('Discarding time.')
        
        def transform_function(index):
            index_list = list(index)
            index_list[0] = 0
            index = tuple(index_list)
            return index
            
        self.transform_indices(transform_function)
    
    
    def discard_space(self):
        logger.debug('Discarding space.')
        
        def transform_function(index):
            index_list = list(index)
            for i in range(1, len(index_list)):
                index_list[i] = 0
                index = tuple(index_list)
            return index
            
        self.transform_indices(transform_function)
    
    
    
    
    
    ## transform results
    
    def transform_result(self, transform_function):
        measurements_dict = self.measurements_dict
        
        for (t, t_dict) in  measurements_dict.items():
            for (x, x_dict) in t_dict.items():
                for (y, y_dict) in x_dict.items():
                    for (z, results) in y_dict.items():
                        results_transformed = transform_function(results)
                        
                        # make to list if not a list
                        try:
                            results_transformed = list(results_transformed)
                        except TypeError:
                            results_transformed = [results_transformed]
                        y_dict[z] = results_transformed
    
    
    
    def normalize(self, same_bounds, minimum_measurements=5):
        logger.debug('Normalizing results with same bounds %s and min measurements %d.' % (same_bounds, minimum_measurements))
        
        ## save measurements dict
        measurements_dict = self.measurements_dict
        
        ## get means and deviations
        self.categorize_indices(same_bounds, discard_year=True)
        means = self.means(minimum_measurements=minimum_measurements, return_type='self')
        deviations = self.deviations(minimum_measurements=minimum_measurements, return_type='self')
        
        ## prepare new measurements dict
        self.measurements_dict = type(measurements_dict)()
        
        ## iterate
        for (t, t_dict) in measurements_dict.items():
            for (x, x_dict) in t_dict.items():
                for (y, y_dict) in x_dict.items():
                    for (z, results_list) in y_dict.items():
                        index = (t, x, y, z)
                        categorized_index = self.categorize_index(index, same_bounds, discard_year=True)
                        
                        try:
                            mean = means.get_value(categorized_index)[0]
                            deviation = deviations.get_value(categorized_index)[0]
                            match = True
                        except KeyError:
                            match = False
                        
                        if match and deviation > 0:
                            for result in results_list:
                                result_normalized = (result - mean) / deviation
                                self.add_result(index, result_normalized)
    
    
    
    ## filter
    def filter(self, filter_function):
        measurements_dict = self.measurements_dict
        measurements_dict_type = type(measurements_dict)
        measurements_dict_transformed = measurements_dict_type()
        self.measurements_dict = measurements_dict_transformed
        
        for (t, t_dict) in  measurements_dict.items():
            for (x, x_dict) in t_dict.items():
                for (y, y_dict) in x_dict.items():
                    for (z, results) in y_dict.items():
                        index = (t, x, y, z)
                        if filter_function(index, results):
                            self.add_result(index, results)
    
    
    def filter_min_measurements(self, min_measurements=1):
        def filter_function(index, results):
            return len(results) >= min_measurements
        
        self.filter(filter_function)
    
    
    
    
    ## compute values
    
    def iterate(self, fun, minimum_measurements=1, return_type='array'):
        measurements_dict = self.measurements_dict
        
        ## check input
        if return_type not in ('array', 'self'):
            raise ValueError('Unknown return_type "%s". Only "array" and "self" are supported.' % return_type)
        
        ## init
        if return_type is 'array':
            values = []
        else:
            values = type(self)()
        
        ## iterate
        for (t, t_dict) in measurements_dict.items():
            for (x, x_dict) in t_dict.items():
                for (y, y_dict) in x_dict.items():
                    for (z, results_list) in y_dict.items():
                        if len(results_list) >= minimum_measurements:
                            results = np.array(results_list)
                            value = fun(results)
                            
                            ## insert
                            if return_type is 'array':
                                row = [t, x, y, z, value]
                                values.append(row)
                            else:
                                index = (t, x, y, z)
                                values.add_result(index, value)
        
        ## finishing
        if return_type is 'array':
            values = np.array(values)
            logger.debug('{} values calculated.'.format(values.shape[0]))
        
        return values
    
    
    
    def numbers(self, minimum_measurements=1, return_type='array'):
        logger.debug('Calculate numbers of measurements with %d minimal measurements.', minimum_measurements)
        
        return self.iterate(len, minimum_measurements, return_type=return_type)
    
    
    def means(self, minimum_measurements=1, return_type='array'):
        logger.debug('Calculate means of measurements with %d minimal measurements.', minimum_measurements)
        
        return self.iterate(np.average, minimum_measurements, return_type=return_type)
    
    
    def variances(self, minimum_measurements=3, min_variance=0, return_type='array'):
        logger.debug('Calculate variances of measurements with %d minimal measurements.', minimum_measurements)
        
        def calculate_variance(results):
            mean = np.average(results)
            number_of_results = results.size
            variance = np.sum((results - mean)**2) / (number_of_results - 1)
            variance = max([variance, min_variance])
            return variance
        
        return self.iterate(calculate_variance, minimum_measurements, return_type=return_type)
    
    
    def deviations(self, minimum_measurements=3, min_deviation=0, return_type='array'):
        logger.debug('Calculate standard deviations of measurements with %d minimal measurements.', minimum_measurements)
        
        def calculate_deviation(results):
            mean = np.average(results)
            number_of_results = results.size
            deviation = (np.sum((results - mean)**2) / (number_of_results - 1))**(1/2)
            deviation = max([deviation, min_deviation])
            return deviation
        
        return self.iterate(calculate_deviation, minimum_measurements, return_type=return_type)
    
    
    
    ## total correlogram and correlation
    
    def _get_first_dim_shifted(self, measurements_dict_list, shift, same_bound, wrap_around_range=None):
        logger.debug('Getting first dim shifted with shift %f and same bound %f.' % (shift, same_bound))
        
        if wrap_around_range is not None:
            wrap_around_len = wrap_around_range[1] - wrap_around_range[0]
        
        shift_list = []
        
        ## iterate over all dicts
        for (measurements_dict, measurements_dict_shifted) in measurements_dict_list:
            
            ## iterate over all unshifted
            for (i, i_dict) in measurements_dict.items():
                i_shifted_desired = i + shift
                if wrap_around_range is not None:
                    i_shifted_desired = measurements.util.calculate.wrap_around_index(i_shifted_desired, wrap_around_range)
                
                ## iterate over all shifted
                for (i_shifted, i_dict_shifted) in measurements_dict_shifted.items():
                    i_diff = abs(i_shifted - i_shifted_desired)
                    if wrap_around_range is not None:
                        i_diff = min(i_diff, wrap_around_len - i_diff)
                    
                    ## insert unshifted in shifted in shift_list
                    if i_diff <= same_bound:
                        shift_list.append((i_dict, i_dict_shifted))
        
        return shift_list
    
    
    
    def _get_results_together_with_shifted(self, factor, direction, same_bounds, x_range, t_range=None):
        logger.debug('Gathering results with direction %s shifted by factor %f with same bound %s.' % (direction, factor, same_bounds))
        
        measurements_dict_list = [(self.measurements_dict, self.measurements_dict)]
        dim = len(direction)
        wrap_around_range = (t_range, x_range, None, None)
        
        ## iterate over dim and search matching shifts
        for i in range(dim):
            measurements_dict_list = self._get_first_dim_shifted(measurements_dict_list, factor * direction[i], same_bounds[i], wrap_around_range=wrap_around_range[i])
        
        logger.debug('Results gathered.')
        
        return measurements_dict_list
    
    
    
    def _get_array_from_shift_list(self, shift_list):
        ## calculate length
#         logger.debug('Getting array from shift list.')
        n = 0
        for (result_list, result_shifted_list) in shift_list:
            n += len(result_list) * len(result_shifted_list)
        
        ## set values
#         logger.debug('Creating array from shift list with length %d.' % n)
        array = np.empty((n, 2))
        i = 0
        for (result_list, result_shifted_list) in shift_list:
            for result in result_list:
                for result_shifted in result_shifted_list:
                    array[i, 0] = result
                    array[i, 1] = result_shifted
                    i += 1
        
        assert i == n
#         if n != i:
#             warning('False array size: n=%d and i=%d' % (n, i))
#         logger.debug('%d elements inserted. Returning array from shift list.' % i)
        
        return array
    
    
    
    def _calculate_total_correlation_from_shift_list(self, shift_list, is_normalized=False):
        if not is_normalized:
            #TODO mean and sd for each result list
            shift_array = self._get_array_from_shift_list(shift_list)
    #         shift_array = np.array(shift_list)
            number = shift_array.shape[0]
            
#             logger.debug('Calulating correlation from %d pairs.' % number)
            
            x = shift_array[:,0]
            y = shift_array[:,1]
            
            mean_x = np.average(x)
            mean_y = np.average(y)
            sd_x = np.sum((x - mean_x)**2)**(1/2)
            sd_y = np.sum((y - mean_y)**2)**(1/2)
            prod_array = ((x - mean_x) * (y - mean_y)) / (sd_x * sd_y)
            
            correlation = np.sum(prod_array)
        else:
            number = 0
            correlation = 0
            
            for (result_list, result_shifted_list) in shift_list:
                for result in result_list:
                    for result_shifted in result_shifted_list:
                        correlation += result * result_shifted
                        number += 1
            
            if number >= 1:
                correlation /= number
            else:
                correlation = np.nan
            
        
        logger.debug('Correlation %f calculated from %d measurements.' % (correlation, number))
        
        return (correlation, number)
    
    
    
    
    
#     def _calculate_correlations_from_shift_list(self, shift_list):
# #         if not is_normalized:
# #             #TODO mean and sd for each result list
# #             shift_array = self._get_array_from_shift_list(shift_list)
# #     #         shift_array = np.array(shift_list)
# #             number = shift_array.shape[0]
# #             
# # #             logger.debug('Calulating correlation from %d pairs.' % number)
# #             
# #             x = shift_array[:,0]
# #             y = shift_array[:,1]
# #             
# #             mean_x = np.average(x)
# #             mean_y = np.average(y)
# #             sd_x = np.sum((x - mean_x)**2)**(1/2)
# #             sd_y = np.sum((y - mean_y)**2)**(1/2)
# #             prod_array = ((x - mean_x) * (y - mean_y)) / (sd_x * sd_y)
# #             
# #             correlation = np.sum(prod_array)
# #         else:
#         number = 0
#         correlation = 0
#         
#         for (result_list, result_shifted_list) in shift_list:
#             xs = np.array(result_list)
#             ys = np.array(result_shifted_list)
#             
#             mean_x = np.average(xs)
#             mean_y = np.average(ys)
#             sd_x = np.sum((xs - mean_x)**2)**(1/2)
#             sd_y = np.sum((ys - mean_y)**2)**(1/2)
#             
#             for x in xs:
#                 for y in ys:
#                     correlation += (x - mean_x) * (y - mean_y)
#                     number += 1
#             prod_array = ((x - mean_x) * (y - mean_y))
#             correlation = np.sum(prod_array) / (sd_x * sd_y)
#         
#         if number >= 1:
#             correlation /= number
#         else:
#             correlation = np.nan
#             
#         
#         logger.debug('Correlation %f calculated from %d measurements.' % (correlation, number))
#         
#         return (correlation, number)
    
    
    
    def _iterate_over_shift_in_direction(self, calculate_function, direction, same_bounds, dim_ranges, wrap_around_t=False, file=None):
        logger.debug('Applying function to shifts by direction %s with same_bounds %s and dim_ranges %s.' % (direction, same_bounds, dim_ranges))
        
        ## init
        function_results_list = []
        direction_array = np.array(direction, dtype=np.float)
        if wrap_around_t:
            t_range = (dim_ranges[0][0], dim_ranges[0][1])
        else:
            t_range = None
        x_range = (dim_ranges[1][0], dim_ranges[1][1])
        
        ## calculate max factor
        if not np.all(direction == 0):
            dim_ranges_array = np.array(dim_ranges, dtype=np.float)
            dim_ranges_diff = dim_ranges_array[:,1] - dim_ranges_array[:,0]
            if wrap_around_t:
                dim_ranges_diff[0] = dim_ranges_diff[0] / 2
            dim_ranges_diff[1] = dim_ranges_diff[1] / 2
            max_factor_mask = direction_array != 0
            max_factor = math.floor(min(dim_ranges_diff[max_factor_mask] / direction_array[max_factor_mask]))
        else:
            max_factor = 0
        
        logger.debug('Max factor is %d.', max_factor)
        
        ## iterate over all factors
        for factor in range(max_factor + 1):
            shift_list = self._get_results_together_with_shifted(factor, direction, same_bounds, x_range, t_range)
            
            ## apply calculate_function to shift list
            logger.debug('Applying calculate function to shifts.')
            function_result = calculate_function(shift_list)
            function_results_list.append(function_result)
            
            ## save intermediate result
            if file is not None:
                function_results_array = np.array(function_results_list)
                np.save(file, function_results_array)
        
        ## create array and save results
        function_results_array = np.array(function_results_list)
        
        if file is None:
            np.save(file, function_results_array)
        
        logger.debug('Results array calculated.')
        
        return function_results_array
    
    
    
    def total_correlogram(self, direction, same_bounds, dim_ranges, wrap_around_t=False, minimum_measurements=1, is_normalized=False, file=None):
        
        logger.debug('Calculating correlogram.')
        
        calculate_correlation = lambda shift_list : self._calculate_total_correlation_from_shift_list(shift_list, is_normalized=is_normalized)
        correlogram = self._iterate_over_shift_in_direction(calculate_correlation, direction, same_bounds, dim_ranges, wrap_around_t=wrap_around_t, file=file)
        
        logger.debug('Correlogram calculated.')
        
        return correlogram
    
    
    
    
    
    def _iterate_over_shift_all_factor_combinations(self, calculation_function, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=1, file=None):
        
        logger.debug('Iterate over all shifts with all factor combinations with the following configurations: direction=%s, factor_lists=%s, same_bounds=%s, wrap_around_ranges=%s, minimum_measurements=%d.' % (direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements))
        
        function_results_list = []
        
        dim = len(direction)
        max_index = np.array([len(factor_list) for factor_list in factor_lists])
        
        measurements_dict_list = [[(self.measurements_dict, self.measurements_dict),],] + [None] * dim
        current_dim = 0
        current_indices = np.zeros(dim, dtype=np.int)
        current_shift = np.zeros(dim)
        
        ## iterate over all factors
        while current_dim >= 0:
            
            ## iterate over all dimensions
            while 0 <= current_dim < dim:
                current_index = current_indices[current_dim]
                current_factor_list = factor_lists[current_dim]
                
                ## search matching shifts
                current_factor = current_factor_list[current_index]
                current_shift[current_dim] = direction[current_dim] * current_factor
                measurements_dict_list[current_dim + 1] = self._get_first_dim_shifted(measurements_dict_list[current_dim], current_shift[current_dim], same_bounds[current_dim], wrap_around_ranges[current_dim])
                
                ## increase current dim
                current_dim += 1
            
            
            ## calculate value and append to list
#             results_with_shifted_list = [(result, result_shifted) for (result_list, result_shifted_list) in measurements_dict_list[dim] for result in result_list for result_shifted in result_shifted_list]
#             results_with_shifted_iterable = itertools.chain([itertools.product(result_list, result_shifted_list) for (result_list, result_shifted_list) in measurements_dict_list[dim]])
            
#             number_of_measurements = len(results_with_shifted_list)
#             (value, number_of_measurements) = calculation_function(results_with_shifted_iterable)
            (value, number_of_measurements) = calculation_function(measurements_dict_list[dim])
            if number_of_measurements >= minimum_measurements:
#                 (value, number_of_measurements) = calculation_function(results_with_shifted_list)
                function_results_list.append(current_shift.tolist() + [value, number_of_measurements])
                logger.debug('Value %f for shift %s calculated and inserted. %d matching measurements where found.' % (value, current_shift, number_of_measurements))
                
                ## save intermediate result
                if file is not None:
                    function_results_array = np.array(function_results_list)
                    np.save(file, function_results_array)
            else:
                logger.debug('Value for shift %s not inserted. Only %d matching measurements where found.' % (current_shift, number_of_measurements))
            
            
            ## increase index
            current_dim -= 1
            measurements_dict_list[current_dim + 1] = None
            current_indices[current_dim] += 1
            
            while current_dim >= 0 and current_indices[current_dim] == len(factor_lists[current_dim]):
                measurements_dict_list[current_dim] = None
                current_indices[current_dim] = 0
                current_dim -= 1
                if current_dim >= 0:
                    current_indices[current_dim] += 1
        
        ## create array and save results
        function_results_array = np.array(function_results_list)
        
        if file is None:
            np.save(file, function_results_array)
        
        logger.debug('Iterate over all shifts with all factor combinations finished.')
        
        return function_results_array
    
    
    def total_correlation(self, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=1, is_normalized=False, file=None):
        logger.debug('Calculating correlation with the following configurations: direction=%s, factor_lists=%s, same_bounds=%s, wrap_around_ranges=%s, minimum_measurements=%d.' % (direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements))
        
        calculate_correlation = lambda shift_list : self._calculate_total_correlation_from_shift_list(shift_list, is_normalized=is_normalized)
        correlation = self._iterate_over_shift_all_factor_combinations(calculate_correlation, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=minimum_measurements, file=file)
        
        logger.debug('Correlation calculated.')
        
        return correlation
    
    



class Measurements_Sorted(Measurements_Unsorted):
    
    def __init__(self):
        from blist import sorteddict
        self.measurements_dict = sorteddict()
    
    
    def _get_first_dim_shifted(self, measurements_dict_list, shift, same_bound, wrap_around_range=None):
        logger.debug('Getting first dim shifted with shift %f and same bound %f.' % (shift, same_bound))
        
        if wrap_around_range is not None:
            wrap_around_len = wrap_around_range[1] - wrap_around_range[0]
        
        shift_list = []
        
        ## iterate over all dicts
        for (measurements_dict, measurements_dict_shifted) in measurements_dict_list:
            
            keys_view_shifted = measurements_dict_shifted.keys()
            keys_view_shifted_len = len(keys_view_shifted)
            
            ## iterate over all unshifted
            for (key, value) in measurements_dict.items():
                
                ## calculate desired key bounds
                key_shifted_desired_lower_bound = measurements.util.calculate.wrap_around_index(key + shift - same_bound, wrap_around_range)
                key_shifted_desired_upper_bound = measurements.util.calculate.wrap_around_index(key + shift + same_bound, wrap_around_range)
                key_shifted_desired_lower_bound_index = keys_view_shifted.bisect_left(key_shifted_desired_lower_bound)
                key_shifted_desired_upper_bound_index = keys_view_shifted.bisect_right(key_shifted_desired_upper_bound)
                
#                 logger.log(logging.DEBUG + 10, 'Getting shifts for key %f with lower bound %f and upper bound %f.' % (key, key_shifted_desired_lower_bound, key_shifted_desired_upper_bound))
                
                ## if desired keys are available
                if key_shifted_desired_lower_bound_index != key_shifted_desired_upper_bound_index:
                    if key_shifted_desired_upper_bound_index >= keys_view_shifted_len or keys_view_shifted[key_shifted_desired_upper_bound_index] > key_shifted_desired_upper_bound:
                        key_shifted_desired_upper_bound_index -= 1
                    
                    ## calculate desired key range
                    if key_shifted_desired_lower_bound_index <= key_shifted_desired_upper_bound_index:
                        key_shifted_desired_index_range = range(key_shifted_desired_lower_bound_index, key_shifted_desired_upper_bound_index + 1)
                    else:
                        key_shifted_desired_index_range = itertools.chain(range(key_shifted_desired_lower_bound_index, keys_view_shifted_len), range(0, key_shifted_desired_upper_bound_index + 1))
                    
                    ## insert values with shifted values
                    for key_shifted_desired_index in key_shifted_desired_index_range:
                        key_shifted = keys_view_shifted[key_shifted_desired_index]
                        value_shifted = measurements_dict_shifted[key_shifted]
                        shift_list.append((value, value_shifted))
                        
#                         logger.log(logging.DEBUG + 10, 'Value for key %f inserted.' % (key_shifted))
#                 else:
#                     logger.log(logging.DEBUG + 10, 'No values in key range found.')
        
        return shift_list
    
    
    
    def save(self, file):
        super().save(file)
    
    
    def load(self, file):
        super().load(file)

