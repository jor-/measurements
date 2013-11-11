import numpy as np
import math
import bisect
import logging

import util.io
import measurements.util


class Measurements():
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.measurements_dict = dict()
    
    
    def add_result(self, t, x, y, z, result):
        dictionary = self.measurements_dict
        index = (t, x, y, z)
        n = len(index)
        for i in range(n-1):
            dictionary = dictionary.setdefault(index[i], dict())
        result_list = dictionary.setdefault(index[n-1], [])
        try:
            result_list.extend(result)
        except TypeError:
            result_list.append(result)
    
    
    def add_cruises_with_box_indices(self, cruises):
        measurements_dict = self.measurements_dict
        
        ## insert results in dict
        for cruise in cruises:
            spatial_indices = cruise.spatial_indices
            t = cruise.dt_float
            results = cruise.po4.astype(float)
            
            for i in range(results.size):
                (x, y, z) = spatial_indices[i]
                self.add_result(t, x, y, z, results[i])
    
    
    def add_cruises_with_coordinates(self, cruises):
        measurements_dict = self.measurements_dict
        
        ## insert results in dict
        for cruise in cruises:
            x = cruise.x
            y = cruise.y
            z = cruise.z
            t = cruise.dt_float
            results = cruise.po4.astype(float)
            
            for i in range(results.size):
                self.add_result(t, x, y, z[i], results[i])
    
    
    
    def save(self, file):
        self.logger.debug('Saving measurements at %s.', file)
        util.io.save_object(self.measurements_dict, file)
    
    
    def load(self, file):
        self.logger.debug('Loading measurements from %s.', file)
        self.measurements_dict = util.io.load_object(file)
    
    
    
    def transform_indices(self, transform_function):
        measurements_dict = self.measurements_dict
        measurements_dict_transformed = dict()
        
        for (t, t_dict) in  measurements_dict.items():
            for (x, x_dict) in t_dict.items():
                for (y, y_dict) in x_dict.items():
                    for (z, results) in y_dict.items():
                        index = (t, x, y, z)
                        index_transformed = transform_function(index)
                        
                        ## insert results in transformed indices
                        n = len(index)
                        dictionary = measurements_dict_transformed
                        for i in range(n-1):
                            dictionary = dictionary.setdefault(index_transformed[i], dict())
                        results_transformed = dictionary.setdefault(index_transformed[n-1], [])
                        results_transformed.extend(results)
                        dictionary[index_transformed[n-1]] = results_transformed
        
        self.measurements_dict = measurements_dict_transformed
    
    
    def categorize_indices(self, separation_values, t_range=None, x_range=None):
        self.logger.debug('Categorize indices by separation values %s.', separation_values)
        
        def transform_function(index):
            index_list = list(index)
            for i in range(len(separation_values)):
                ## categorize dimension i
                separation_values_i = separation_values[i]
                if separation_values_i is not None:
                    # check if sequence
                    try:
                        is_sequence = len(separation_values_i) >= 2
                    except TypeError:
                        is_sequence = False
                    
                    # separate by value
                    if not is_sequence:
                        index_list[i] = (math.floor(index[i] / separation_values_i) + 0.5) * separation_values_i
                    # separate by sequence
                    else:
                        index_right = bisect.bisect(separation_values_i, index[i])
                        value_left = separation_values_i[index_right - 1]
                        try:
                            value_right = separation_values_i[index_right]
                        except IndexError:
                            raise ValueError('Index value %d exceeds range of separation values (right sight of separation values is %d).' % (index[i], value_left))
                        index_list[i] = (value_left + value_right) / 2.0
                    
                    # wrap around t
                    if i==0 and t_range is not None:
                        index_list[i] = measurements.util.wrap_around_index(index_list[i], t_range)
                    # wrap around x
                    if i==1 and x_range is not None:
                        index_list[i] = measurements.util.wrap_around_index(index_list[i], x_range)
            
            index = tuple(index_list)
            return index
            
        self.transform_indices(transform_function)
    
    
    def discard_year(self):
        self.logger.debug('Discarding year.')
        
        def transform_function(index):
            index_list = list(index)
            index_list[0] = index[0] % 1
            index = tuple(index_list)
            return index
            
        self.transform_indices(transform_function)
    
    
    def discard_time(self):
        self.logger.debug('Discarding time.')
        
        def transform_function(index):
            index_list = list(index)
            index_list[0] = 0
            index = tuple(index_list)
            return index
            
        self.transform_indices(transform_function)
    
    
    
    
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
    
    
    
#     #TODO: rewrite (new dic struct)
#     def filter_by_space(self, filter_function):
#         measurements_dict = self.measurements_dict
#         filtered_measurements_dict = dict()
#         
#         for (space_index, time_dict) in measurements_dict.items():
#             if filter_function(space_index):
#                 filtered_measurements_dict[space_index] = time_dict
#         
#         filtered_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
#         filtered_measurements.measurements_dict = filtered_measurements_dict
#         
#         return filtered_measurements
#     
#     
#     #TODO: rewrite (new dic struct)
#     def filter_space_values(self, x=None, y=None, z=None):
#         compare_value_function = lambda value1, value2: value1 is None or value2 is None or value1 == value2 
#         filter_function = lambda space_index: compare_value_function(x, space_index[0]) and compare_value_function(y, space_index[1]) and compare_value_function(z, space_index[2])
#         
#         return self.filter_by_space(filter_function)
#     
#     
#     #TODO: rewrite (new dic struct)
#     def filter_by_time(self, filter_function):
#         measurements_dict = self.measurements_dict
#         filtered_measurements_dict = dict()
#         
#         for (space_index, time_dict) in measurements_dict.items():
#             for (time_index, results_list) in time_dict.items():
#                 if filter_function(time_index):
#                     time_dict = filtered_measurements_dict.setdefault(space_index, dict())
#                     time_dict[time_index] = results_list
#         
#         filtered_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
#         filtered_measurements.measurements_dict = filtered_measurements_dict
#         
#         return filtered_measurements
#     
#     
#     #TODO: rewrite (new dic struct)
#     def filter_time_range(self, lower_bound, upper_bound):
#         filter_function = lambda time: time >= lower_bound and time <= upper_bound
#         
#         return self.filter_by_time(filter_function)
    
    
    
    
    def iterate(self, fun, minimum_measurements=1, return_as_map=False, map_default_value=float('nan')):
        measurements_dict = self.measurements_dict
        value_list = []
        
        for (t, t_dict) in measurements_dict.items():
            for (x, x_dict) in t_dict.items():
                for (y, y_dict) in x_dict.items():
                    for (z, results_list) in y_dict.items():
                        if len(results_list) >= minimum_measurements:
                            results = np.array(results_list)
                            value = fun(results)
                            row = [t, x, y, z, value]
                            
                            value_list.append(row)
        
        values = np.array(value_list)
        
        if return_as_map:
            values = measurements.util.insert_values_in_map(values, default_value=map_default_value)
        
        return values
    
    
    
    def number_of_measurements(self, minimum_measurements=1, return_as_map=False):
        self.logger.debug('Calculate numbers of measurements with %d minimal measurements.', minimum_measurements)
        
        return self.iterate(len, minimum_measurements, return_as_map, map_default_value=0)
    
    
    def means(self, minimum_measurements=1, return_as_map=False):
        self.logger.debug('Calculate means of measurements with %d minimal measurements.', minimum_measurements)
        
        return self.iterate(np.average, minimum_measurements, return_as_map)
    
    
    def variances(self, minimum_measurements=3, return_as_map=False):
        self.logger.debug('Calculate variances of measurements with %d minimal measurements.', minimum_measurements)
        
        def calculate_variance(results):
            mean = np.average(results)
            number_of_results = results.size
            variance = np.sum((results - mean)**2) / (number_of_results - 1)
            return variance
        
        return self.iterate(calculate_variance, minimum_measurements, return_as_map, map_default_value=float('inf'))
    
    
    def standard_deviations(self, minimum_measurements=3, return_as_map=False):
        self.logger.debug('Calculate standard deviations of measurements with %d minimal measurements.', minimum_measurements)
        
        def calculate_standard_deviation(results):
            mean = np.average(results)
            number_of_results = results.size
            standard_deviation = (np.sum((results - mean)**2) / (number_of_results - 1))**(1/2)
            return standard_deviation
        
        return self.iterate(calculate_standard_deviation, minimum_measurements, return_as_map, map_default_value=float('inf'))
    
    
    
    def get_first_dim_shifted(self, measurements_dict_list, shift, same_bound, wrap_around_range=None):
        if wrap_around_range is not None:
            wrap_around_len = wrap_around_range[1] - wrap_around_range[0]
        
        shift_list = []
        
        ## iterate over all dicts
        for (measurements_dict, measurements_dict_shifted) in measurements_dict_list:
            
            ## iterate over all unshifted
            for (i, i_dict) in measurements_dict.items():
                i_shifted_desired = i + shift
                if wrap_around_range is not None:
                    i_shifted_desired = measurements.util.wrap_around_index(i_shifted_desired, wrap_around_range)
                
                ## iterate over all shifted
                for (i_shifted, i_dict_shifted) in measurements_dict.items():
                    i_diff = abs(i_shifted - i_shifted_desired)
                    if wrap_around_range is not None:
                        i_diff = min(i_diff, wrap_around_len - i_diff)
                    
                    ## insert unshifted in shifted in shift_list
                    if i_diff <= same_bound:
                        shift_list.append((i_dict, i_dict_shifted))
        
        return shift_list
    
    
    def iterate_over_shift_all_factor_combinations(self, calculation_function, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=1):
        
        self.logger.debug('Iterate over all shifts with all factor combinations with the following configurations: direction=%s, factor_lists=%s, same_bounds=%s, wrap_around_ranges=%s, minimum_measurements=%s.' % (direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements))
        
        value_list = []
        
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
                measurements_dict_list[current_dim + 1] = self.get_first_dim_shifted(measurements_dict_list[current_dim], current_shift[current_dim], same_bounds[current_dim], wrap_around_ranges[current_dim])
                
                ## increase current dim
                current_dim += 1
            
            
            ## calculate value and append to list
            results_with_shifted_list = [(result, result_shifted) for (result_list, result_shifted_list) in measurements_dict_list[dim] for result in result_list for result_shifted in result_shifted_list]
            number_of_measurements = len(results_with_shifted_list)
            if number_of_measurements >= minimum_measurements:
                value = calculation_function(results_with_shifted_list)
                value_list.append(current_shift.tolist() + [value, number_of_measurements])
                self.logger.debug('Value %d for shift %s calculated and inserted. %d matching measurements where found.' % (value, current_shift, n))
            else:
                self.logger.debug('Value for shift %s not inserted. Only %d matching measurements where found.' % (current_shift, n))
            
            
            ## increase index
            current_dim -= 1
            current_indices[current_dim] += 1
            
            while current_dim >= 0 and current_indices[current_dim] == len(factor_lists[current_dim]):
                current_indices[current_dim] = 0
                current_dim -= 1
                if current_dim >= 0:
                    current_indices[current_dim] += 1
        
        
        value_array = np.array(value_list)
        
        self.logger.debug('Iterate over all shifts with all factor combinations finished.')
        
        return value_array
    
    
    def correlation(self, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=1):
        def calculate_correlation(shift_list):
            shift_array = np.array(shift_list)
            x = shift_array[:,0]
            y = shift_array[:,1]
            
            mean_x = np.average(x)
            mean_y = np.average(y)
            sd_x = (np.sum((x - mean_x)**2) / number)**(1/2)
            sd_y = (np.sum((y - mean_y)**2) / number)**(1/2)
            prod_array = ((x - mean_x) * (y - mean_y)) / (sd_x * sd_y)
            correlation = np.sum(prod_array) / number
            
            return correlation
        
        
        self.logger.debug('Calculating correlation with the following configurations: direction=%s, factor_lists=%s, same_bounds=%s, wrap_around_ranges=%s, minimum_measurements=%s.' % (direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements))
        
        correlation = self.iterate_over_shift_all_factor_combinations(calculate_correlation, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=minimum_measurements)
        
        self.logger.debug('Correlation calculated.')
        
        return correlation
    
    
    
    def get_results_together_with_shifted(self, factor, direction, same_bound, x_range, t_range=None):
        self.logger.debug('Gathering results with direction %s shifted by factor %d.', direction, factor)
        
        measurements_dict_list = [[(self.measurements_dict, self.measurements_dict)]]
        dim = len(direction)
        wrap_around_range = (t_range, x_range, None, None)
        
        ## iterate over dim and search matching shifts
        for i in range(dim):
            measurements_dict_list = self.get_first_dim_shifted(measurements_dict_list, factor * direction[i], same_bound[i], wrap_around_range=wrap_around_range[i])
        
        ## create list of results together with shifted results
        results_with_shifted_list = [(result, result_shifted) for (result_list, result_shifted_list) in measurements_dict_list for result in result_list for result_shifted in result_shifted_list]
        
#         measurements_dict = self.measurements_dict
#         shift_list = []
#         
#         
#         ## iterate over t
#         for (t, t_dict) in measurements_dict.items():
#             t_shifted_desired = t + factor * direction[0]
#             t_shifted_desired = measurements.util.wrap_around_index(t_shifted_desired, t_range)
#             for (t_shifted, t_dict_shifted) in measurements_dict.items():
#                 t_diff = t_shifted - t_shifted_desired
#                 if abs(t_diff) <= same_bound[0]:
#                     
#                     ## iterate over x
#                     for (x, x_dict) in t_dict.items():
#                         x_shifted_desired = x + factor * direction[1]
#                         x_shifted_desired = measurements.util.wrap_around_index(x_shifted_desired, x_range)
#                         for (x_shifted, x_dict_shifted) in t_dict_shifted.items():
#                             x_diff = x_shifted - x_shifted_desired
# #                             x_diff = wrap_around_x(x_diff)
#                             if abs(x_diff) <= same_bound[1]:
#                                 
#                                 ## iterate over y
#                                 for (y, y_dict) in x_dict.items():
#                                     y_shifted_desired = y + factor * direction[2]
#                                     for (y_shifted, y_dict_shifted) in x_dict_shifted.items():
#                                         y_diff = y_shifted - y_shifted_desired
#                                         if abs(y_diff) <= same_bound[2]:
#                                             
#                                             ## iterate over z
#                                             for (z, results_list) in y_dict.items():
#                                                 z_shifted_desired = z + factor * direction[3]
#                                                 for (z_shifted, results_list_shifted) in y_dict_shifted.items():
#                                                     z_diff = z_shifted - z_shifted_desired
#                                                     if abs(z_diff) <= same_bound[3]:
#                                                         self.logger.log(logging.DEBUG - 10, 'Current index is: %s. Shifted index is: %s.', (t, x, y, z), (t_shifted, x_shifted, y_shifted, z_shifted))
#                                                         
#                                                         ## insert shift to shift list
#                                                         for result in results_list:
#                                                             for result_shifted in results_list_shifted:
#                                                                 shift_list.append((result, result_shifted))
        
        
        self.logger.debug('Results gathered.')
        
        return results_with_shifted_list
        
    
    def iterate_over_shift_in_direction(self, calculate_function, direction, same_bound, dim_ranges, wrap_around_t=False, file=None):
        self.logger.debug('Applying function to shifts by direction %s.', direction)
        
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
        
        self.logger.debug('Max factor is %d.', max_factor)
        
        ## iterate over all factors
        for factor in range(max_factor + 1):
            shift_list = self.get_results_together_with_shifted(factor, direction, same_bound, x_range, t_range)
            
            ## apply calculate_function to shift list
            self.logger.debug('Applying calculate function to %d shifts.', len(shift_list))
            function_result = calculate_function(shift_list)
            function_results_list.append(function_result)
            
            ## save intermediate result
            if file is not None:
                function_results_array = np.array(function_results_list)
                np.save(file, function_results_array)
        
        if file is None:
            function_results_array = np.array(function_results_list)
        
        self.logger.debug('Results array calculated.')
        
        return function_results_array
    
    
    
    def correlogram(self, direction, same_bound, dim_ranges, wrap_around_t=False, minimum_measurements=1, file=None):
        def calculate_function(shift_list):
            number = len(shift_list)
            
            if number >= minimum_measurements:
                shift_array = np.array(shift_list)
                x = shift_array[:,0]
                y = shift_array[:,1]
                mean_x = np.average(x)
                mean_y = np.average(y)
                sd_x = (np.sum((x - mean_x)**2) / number)**(1/2)
                sd_y = (np.sum((y - mean_y)**2) / number)**(1/2)
                prod_array = ((x - mean_x) * (y - mean_y)) / (sd_x * sd_y)
                correlation = np.sum(prod_array) / number
                percentiles = np.percentile(prod_array, (5, 50, 95), overwrite_input=True)
                
                result = (correlation,) + tuple(percentiles) + (number,)
            else:
                result = (float('nan'),)*4 + (number,)
            
            return result
        
        
        self.logger.debug('Calculating correlogram.')
        
        correlogram = self.iterate_over_shift_in_direction(calculate_function, direction, same_bound, dim_ranges, wrap_around_t=wrap_around_t, file=file)
        
        self.logger.debug('Correlogram calculated.')
        
        return correlogram
        
    
    
    
# #     def calculate_from_increment(self, calculate_function, direction=(0, 0, 0, 0)):
# #         from .constants import SAME_DATETIME_BOUND
# #         from ndop.metos3d.constants import METOS_DIM
# #         
# #         self.print_debug_inc(('Calculating results list from increments for direction ', direction, '.'))
# #         
# #         measurements_dict = self.measurements_dict
# #         function_results_list = []
# #         
# #         ## prepare directions
# #         direction_array = np.array(direction)
# #         space_direction_array = direction_array[0:-1]
# #         time_direction = direction[-1]
# #         
# #         ## calculate max factor
# #         if np.all(direction == 0):
# #             max_factor = 0
# #         else:
# #             max_factor_dim = np.array(METOS_DIM + (100,0))
# #             max_factor_dim[0:2] = max_factor_dim[0:2] / 2
# #             max_factor_mask = direction_array != 0
# #             max_factor = math.floor(min(max_factor_dim[max_factor_mask] / direction_array[max_factor_mask]))
# #         
# #         self.print_debug(('Max factor is ', max_factor, '.'))
# #         
# #         ## iterate over all factors
# #         for factor in range(max_factor + 1):
# #             self.print_debug_inc(('Using factor ', factor, '.'))
# #             
# #             increment_list = []
# #                 
# #             ## iterate over all measurements
# #             for (space_index, time_dict) in measurements_dict.items():
# #                 self.print_debug_inc(('Looking at space index ', space_index, '.'))
# #                 space_index_array = np.array(space_index)
# #                 
# #                 ## calculate space incremented index
# #                 space_incremented_index_array = space_index_array + factor * space_direction_array
# #                 for i in range(2):
# #                     if space_incremented_index_array[i] < 0:
# #                         space_incremented_index_array[i] += METOS_DIM[i]
# #                     elif space_incremented_index_array[i] >= METOS_DIM[i]:
# #                         space_incremented_index_array[i] -= METOS_DIM[i]
# #                 space_incremented_index = tuple(space_incremented_index_array)
# #                 
# #                 self.print_debug(('Space incremented index is ', space_incremented_index, '.'))
# #                 
# #                 ## get time dict for space incremented index 
# #                 try:
# #                     time_incremented_dict = measurements_dict[space_incremented_index]
# #                 except KeyError:
# #                     time_incremented_dict = None
# #                 
# #                 
# #                 ## iterate over all time combinations
# #                 if time_incremented_dict is not None:
# #                     for (time_index, results_list) in time_dict.items():
# #                         time_incremented_index_desired = time_index + factor * time_direction
# #                         for (time_incremented_index, results_incremented_list) in time_incremented_dict.items():
# #                             time_diff = time_incremented_index - time_incremented_index_desired
# #                             
# #                             ## insert increment to increment list if desired time diff
# #                             if time_diff >= 0 and time_diff <= SAME_DATETIME_BOUND:
# #                                 for result in results_list:
# #                                     for result_incremented in results_incremented_list:
# #                                         increment= (result_incremented - result)**2
# #                                         increment_list.append(increment)
# #                 
# #                 self.required_debug_level_dec()
# #             
# #             ## apply calculate_function to increment list
# #             self.print_debug(('Applying calculate function to ', len(increment_list), ' increments.'))
# #             
# #             function_result = calculate_function(increment_list)
# #             function_results_list.append(function_result)
# #             
# #             self.required_debug_level_dec()
# #             
# #         self.print_debug_dec('Results list calculated.')
# #         
# #         return function_results_list
#     
#     
#     #TODO: rewrite to use iterate_over_shift_in_direction
#     def variogram(self, direction=(0, 0, 0, 0)):
#         def calculate_function(increment_list):
#             number = len(increment_list)
#             
#             if number > 0:
#                 increment_array = np.array(increment_list) / 2
#                 mean = np.average(increment_array)
#                 percentiles = np.percentile(increment_array, (2.5, 50, 97.5), overwrite_input=True)
#                 
#                 result = (mean,) + tuple(percentiles) + (number,)
#             else:
#                 result = (float('nan'),)*4 + (0,)
#             
#             return result
#         
#         
#         self.print_debug('Calculating variogram list.')
#         
#         variogram_list = self.calculate_from_increment(calculate_function, direction)
#         
#         
# #         variogram_list = []
# #         for (space_factor, variogram_time_dict) in variogram_dict.items():
# #             for (time_factor, variogram_result) in variogram_time_dict.items():
# #                 variogram_result = (space_factor, time_factor) + variogram_result
# #                 variogram_list.append(variogram_result)
#         
#         self.print_debug('Calculating variogram array.')
#         variogram = np.array(variogram_list)
#         
#         self.print_debug_dec('Variogram calculated.')
#         
#         return variogram
#         
#         ##################
#         #TODO: plot (averaged) increment per space
#         #TODO: discard (different) axis at measurements
#         #TODO: transform at all axis possible
#         ##################
#     
#     
#     
#     
# #     def variogram(self, space_offset=(0, 0, 0), time_offset=0, minimum_measurements=50):
# #         from .constants import SAME_DATETIME_BOUND
# #         from ndop.metos3d.constants import METOS_X_DIM, METOS_Y_DIM
# #         
# #         self.print_debug_inc('Calculating variogram.')
# #         
# #         transform_time_function = lambda t: math.floor(t / SAME_DATETIME_BOUND) * SAME_DATETIME_BOUND
# #         measurements_dict = self.transform_time(transform_time_function).measurements_dict
# #         variogram_dict = dict()
# #         
# #         space_offset_array = np.array(space_offset)
# #         if np.all(space_offset_array == 0):
# #             max_space_factor = 1
# #         else:
# #             max_space_factor = METOS_X_DIM
# #         
# #         ## compute variogram_dict
# #         self.print_debug('Calculating variogram dict.')
# #         
# #         ## iterate over all measurements
# #         for (space_index, time_dict) in measurements_dict.items():
# #             self.print_debug_inc(('Looking at space index ', space_index, '.'))
# #             space_index_array = np.array(space_index)
# #             
# #             ## iterate over all possible space factors
# #             for space_factor in range(max_space_factor):
# #                 self.print_debug_inc(('Using space factor ', space_factor, '.'))
# #                 
# #                 ## calculate space offset
# #                 space_offset_index_array = space_index_array + space_factor * space_offset_array
# #                 if space_offset_index_array[0] < 0:
# #                     space_offset_index_array[0] += METOS_X_DIM
# #                 if space_offset_index_array[1] < 0:
# #                     space_offset_index_array[1] += METOS_Y_DIM
# #                 space_offset_index = tuple(space_offset_index_array)
# #             
# #                 ## get time dict for space offset 
# #                 try:
# #                     time_offset_dict = measurements_dict[space_offset_index]
# #                 except KeyError:
# #                     time_offset_dict = None
# #                 
# #                 ## iterate over all time combinations
# #                 if time_offset_dict is not None:
# #                     variogram_time_dict = variogram_dict.setdefault(space_factor, dict())
# #                     
# #                     for (time_index, results_list) in time_dict.items():
# #                         for (time_offset_index, results_offset_list) in time_offset_dict.items():
# #                             
# #                             if time_offset == 0:
# #                                 time_factor = 0
# #                             else:
# #                                 time_factor = round((time_index - time_offset_index) / time_offset)
# #                             
# #                             ## insert results in varigram dict
# #                             if time_offset != 0 or time_index == time_offset_index:
# #                                 
# #                                 (variogram_sum, variogram_number) = variogram_time_dict.setdefault(time_factor, (0, 0))
# #                                 
# #                                 for result in results_list:
# #                                     for result_offset in results_offset_list:
# #                                         variogram_sum += (result - result_offset)**2
# #                                 variogram_number += len(results_list) * len(results_offset_list)
# #                                 
# #                                 variogram_time_dict[time_factor] = (variogram_sum, variogram_number)
# #                     
# #                     
# #                     self.print_debug_inc_dec((variogram_number, ' measurement combinations used.'))
# #                 
# #                 self.required_debug_level_dec()
# #             
# #             self.required_debug_level_dec()
# #         
# #         
# #         ## compute variogram list
# #         self.print_debug('Calculating variogram list.')
# #         
# #         variogram_list = []
# #         for (space_factor, variogram_time_dict) in variogram_dict.items():
# #             for (time_factor, variogram_result) in variogram_time_dict.items():
# #                 (variogram_sum, variogram_number) = variogram_result
# #                 
# #                 if variogram_number >= minimum_measurements:
# #                     variogram_result = variogram_sum / (2 * variogram_number) 
# #                     variogram_list.append((space_factor, time_factor, variogram_result))
# #         
# #         
# #         self.print_debug('Calculating variogram array.')
# #         variogram = np.array(variogram_list)
# #         
# #         self.print_debug_dec('Variogram calculated.')
# #         
# #         return variogram
# #         
# #         
# #             
# #             #####################
# #             for (time_index, results_list) in time_dict.items():
# #                 
# #                 space_offset_index = (space_index[0] + space_offset[0], space_index[1] + space_offset[1], space_index[2] + space_offset[2])
# #                 time_offset_index = time_index + time_offset
# #                 
# #                 
# #                 
# #                 ## get results for time offset
# #                 try:
# #                     results_offset_list = time_offset_dict[time_offset_index]
# #                 except KeyError:
# #                     results_offset_list = []
# #                 
# #                 
# # #                 for (time_index_spatial_offset, results_offset_list) in time_offset_dict.items():
# # #                     if abs(time_index - time_index_spatial_offset) <= SAME_DATETIME_BOUND:
# #                         
# #                 
# #                 
# #                 
# #                 if len(results_list) >= minimum_measurements:
# #                     index = space_index + (time_index,)
# #                     results = np.array(results_list)
# #         
# #         
# #         measurements_dict = self.measurements_dict
# #         variogram_measurements_dict = dict()
# #         
# #         for (space_index, time_dict) in measurements_dict.items():
# #             variogram_time_dict = variogram_measurements_dict.setdefault(space_index, dict())
# #             
# #             for (time_index, results_list) in time_dict.items():
# #                 variogram_time_index = transform_function(time_index)
# #                 variogram_result_list = variogram_time_dict.setdefault(variogram_time_index, [])
# #                 variogram_result_list += results_list
# #         
# #         variogram_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
# #         variogram_measurements.measurements_dict = variogram_measurements_dict
# #         
# #         return variogram_measurements
#         
#     
#     
# #     def get_correlation(self, time_offset=0, spatial_offset=(0,0,0), minimum_measurements=3):
# #         from .constants import SAME_DATETIME_BOUND
# #         measurements_dict = self.measurements_dict
# #         spatial_indices_list = []
# #         value_list = []
# #         
# #         for (space_index, time_results_list) in measurements_dict.items:
# #             if len(time_results_list) >= minimum_measurements:
# #                 space_index_offset = tuple(map(sum, zip(space_index, spatial_offset)))
# #                 time_results_list_offset = measurements_dict[space_index_offset]
# #                 if len(time_results_list_offset) >= minimum_measurements:
# #                     matching_results_list = []
# #                     
# #                     ## find matching results
# #                     for i in range(len(time_results_list)):
# #                         time_i, result_i = time_results_list[i]
# #                         for j in range(len(time_results_list_offset)):
# #                             time_j, result_j = time_results_list_offset[j]
# #                             
# #                             if abs(time_i - time_j - time_offset) < SAME_DATETIME_BOUND
# #                                 matching_results_list.append((result_i, result_j))
# #                     
# #                     ## if enough matching results, calculate covariance
# #                     if len(matching_results_list) >= minimum_measurements:
# #                         spatial_indices_list.append(space_index)
# #                 
# #                 
# #                 spatial_indices_list.append(space_index)
# #                 
# #                 results = np.array(results)[1].astype(float)
# #                 value = fun(results)
# #                 
# #                 value_list.append(value)
# #         
# #         spatial_indices = np.array(spatial_indices_list, dtype=np.uint16)
# #         values = np.array(value_list)
# #         
# #         return (spatial_indices, values)
#         
#         
#         
# # #         measurement_results_dict = self.measurement_results
# # #         measurements_list = []
# # #         value_list = []
# # #         
# # #         for (measurement, results) in measurement_results_dict.items:
# # #             if len(results) >= minimum_measurements:
# # #                 tuple(map(sum, zip((1, 2), (3, 4))))
# # #                 
# # #                 measurements_list.append(measurement)
# # #                 
# # #                 results = np.array(results)[1].astype(float)
# # #                 value = fun(results)
# # #                 
# # #                 value_list.append(value)
# # #                 
# # #                 tuple(map(sum, zip((1, 2), (3, 4))))
# # #         
# # #         measurements = np.array(measurements_list, dtype=np.uint16)
# # #         values = np.array(value_list)
# # #     
# # # #     def get_means(self, minimum_measurements=1):
# # # #         measurement_results_dict = self.measurement_results
# # # #         measurements_list = []
# # # #         means_list = []
# # # #         
# # # #         for (measurement, results) in measurement_results_dict.values:
# # # #             if len(results) >= minimum_measurements:
# # # #                 measurements_list.append(measurement)
# # # #                 
# # # #                 results = np.array(results)
# # # #                 mean = np.average(results)
# # # #                 
# # # #                 means_list.append(mean)
# # # #         
# # # #         measurements = np.array(measurements_list)
# # # #         means = np.array(means_list)
# # # #         
# # # #         return (measurements, means)
# # # #     
# # # #     
# # # #     def get_standard_deviations(self, minimum_measurements=3):
# # # #         measurement_results_dict = self.measurement_results
# # # #         measurements_list = []
# # # #         standard_deviations_list = []
# # # #         
# # # #         for (measurement, results) in measurement_results_dict.values:
# # # #             if len(results) >= minimum_measurements:
# # # #                 measurements_list.append(measurement)
# # # #                 
# # # #                 results = np.array(results)
# # # #                 mean = np.average(results)
# # # #                 number_of_results = results.size
# # # #                 standard_deviation = (np.sum((results - mean)**2) / (number_of_results - 1))**(1/2)
# # # #                 
# # # #                 standard_deviations_list.append(standard_deviation)
# # # #         
# # # #         measurements = np.array(measurements_list)
# # # #         standard_deviations = np.array(standard_deviations_list)
# # # #         
# # # #         return (measurements, standard_deviations)
# # #     
# # #     
# # #     def get_correlation(self, minimum_measurements=3):
# # #         
# # #         measurements, means = self.get_means(minimum_measurements=minimum_measurements)
# # #         
# # #         measurement_results_dict = self.measurement_results
# # #         number_of_measurements = means.size
# # #         
# # #         correlation = np.empty((number_of_measurements, number_of_measurements)) * np.nan
# # #         
# # #         for i in range(number_of_measurements):
# # #             mean_i = means[i]
# # #             for j in range(i+1, number_of_measurements):
# # #                 mean_j = means[j]
# # #                 
# # #             
# # #         
# # #         
# # #         
# # #         measurements_list = []
# # #         means_list = []
# # #         standard_deviations_list = []
# # #         
# # #         for (measurement, results) in measurement_results_dict.values:
# # #             if len(results) >= minimum_measurements:
# # #                 measurements_list.append(measurement)
# # #                 
# # #                 results = np.array(results)
# # #                 mean = np.average(results)
# # #                 means_list.append(mean)
# # #                 standard_deviation = np.average((results - mean)**2)
# # #                 
# # #                 standard_deviations_list.append(standard_deviation)
# # #         
# # #         measurements = np.array(measurements_list)
# # #         standard_deviations = np.array(standard_deviations_list)
# # #         
# # #         return (measurements, standard_deviations)
#     
# #         
# #     
# #     
# #     def get_standard_deviation(self):
# #         
# #     
# #     def get_covariance(self, cruise_collection, minimum_measurements=3):
# #         