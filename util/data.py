import math
import bisect
import itertools

import numpy as np
import scipy.stats


import measurements.util.calculate

import util.logging
import util.multi_dict

logger = util.logging.logger




class Measurements(util.multi_dict.MultiDict):
    
    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)
    
    
    
    ## create
    
    def _return_items_as_type(self, keys, values, return_type=None):
        if return_type == 'measurements' or return_type == 'measurements_unsorted':
            return_type = 'self_type_unsorted'
        if return_type == 'measurements_sorted':
            return_type = 'self_type_sorted'
        return super()._return_items_as_type(keys, values, return_type=return_type)
    
    
    
    ## transform keys
    
    @staticmethod
    def categorize_index(index, separation_values, discard_year=False):
        index = list(index)
        
        ## discard year
        if discard_year:
            index[0] = index[0] % 1
        
        ## remove right bound of last y box
        if index[2] == 90:
            index[2] = 90 - 10**(-6)
        
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
        
        index = tuple(index)
        return index
    
    
    def categorize_indices(self, separation_values, discard_year=False):
        
        
        if discard_year:
            logger.debug('Indices categorized by separation values %s and discard year.' % str(separation_values))
        else:
            logger.debug('Indices categorized by separation values %s.' % str(separation_values))
        
        transform_function = lambda index: self.categorize_index(index, separation_values, discard_year=discard_year)
        
        self.transform_keys(transform_function)
    
    
    def categorize_indices_to_lsm(self, lsm, discard_year=False):
        self.categorize_indices((1/lsm.t_dim, 360/lsm.x_dim, 180/lsm.y_dim, lsm.z), discard_year=discard_year)
    
    
    
    def transform_indices_to_boxes(self, x_dim, y_dim, z_values_left):
        def transform_index_to_boxes(index, x_dim, y_dim, z_values_left):
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
            index[3] = bisect.bisect_right(z_values_left, index[3]) - 1
            
            index = tuple(index)
            
            return index
        
        
        logger.debug('Transform indices to boxes with x_dim {}, y_dim {} and z_values_left {}.'.format(x_dim, y_dim, z_values_left))
        
        transform_function = lambda index: transform_index_to_boxes(index, x_dim=x_dim, y_dim=y_dim, z_values_left=z_values_left)
        
        self.transform_keys(transform_function)
    
    
    def transform_indices_to_lsm(self, lsm):
        def transform_t(index):
            index = list(index)
            index[0] = index[0] % 1
            index[0] = math.floor(index[0] * lsm.t_dim)
            return index
        
        self.transform_keys(transform_t)
        self.transform_indices_to_boxes(lsm.x_dim, lsm.y_dim, lsm.z_left)
    
    
    def coordinates_to_map_indices(self, lsm):
        self.transform_keys(lambda point: lsm.coordinate_to_map_index(*point))
    
    def map_indices_to_coordinates(self, lsm):
        self.transform_keys(lambda index: lsm.map_index_to_coordinate(*index))
    
    
    def discard_year(self):
        logger.debug('Discarding year.')
        
        def transform_function(key):
            key_list = list(key)
            key_list[0] = key[0] % 1
            key = tuple(key_list)
            return key
            
        self.transform_keys(transform_function)
    
    
    def discard_time(self):
        logger.debug('Discarding time.')
        self.dicard_key_dim(0)
    
    def discard_space(self):
        logger.debug('Discarding space.')
        self.dicard_key_dims((1,2,3))
    
    
    
    ## transform values
    
    def normalize(self, same_bounds, min_values=5):
        logger.debug('Normalizing values with same bounds {} and min measurements {}.'.format(same_bounds, min_values))
        
        ## save measurements dict
        value_dict = self.value_dict
        
        ## get means and deviations
        self.categorize_indices(same_bounds, discard_year=True)
        means = self.means(min_values=min_values, return_type='self_type_unsorted')
        deviations = self.deviations(min_values=min_values, return_type='self_type_unsorted')
        
        ## prepare new measurements dict
        self.clear()
        
        ## iterate
        for (key, value_list) in self._iterate_generator_value_dict(value_dict):
            categorized_key = self.categorize_index(key, same_bounds, discard_year=True)
            try:
                mean = means[categorized_key][0]
                deviation = deviations[categorized_key][0]
                match = True
            except KeyError:
                match = False
            
            if match and deviation > 0:
                new_value_list = []
                for value in value_list:
                    value_normalized = (value - mean) / deviation
                    self.append_value(key, value_normalized)
    
    
    def normalize_with_lsm(self, lsm, min_values=5):
        same_bounds = (1/lsm.t_dim, 360/lsm.x_dim, 180/lsm.y_dim, lsm.z)
        return self.normalize(same_bounds, min_values=min_values)
    
    
    
    ## filter
    
    def filter_year(self, year):
        return self.filter_key_range(0, [year, year+1-10**(-10)])
    
    
    def filter_same_point_with_bounds(self, point, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True):
        ## equal_bounds is float -> copy values for each dim
        try:
            float(equal_bounds)
            equal_bounds = [equal_bounds]*4
        except TypeError:
            pass
        
        ## search equal_bound for z if z has sequence
        assert len(equal_bounds) == 4
        equal_bounds = list(equal_bounds)
        try:
            float(equal_bounds[3])
        except TypeError:
            ## get z bound for point z value
            z_bound_found = False
            i = 0
            while not z_bound_found:
                if point[3] <= equal_bounds[3][i+1, 0]:
                    z_bound_found = True
                else:
                    i += 1
                    z_bound_found = i == len(equal_bounds[3]) - 1
            equal_bounds[3] = equal_bounds[3][i, 1]
        equal_bounds = np.array(equal_bounds)
        
        ## prepare point
        point_base = np.array(point, copy=True)
        if discard_year:
            point_base[0] = point_base[0] % 1
        
        
        ## filter all measurements with point in equal bounds
        filtered_points = []
        filtered_results = []
        
        for (t, t_dict) in  self.value_dict.items():
            if (not discard_year and np.abs(point_base[0] - t) <= equal_bounds[0]) or (discard_year and np.abs(point_base[0] - t%1) <= equal_bounds[0]):
                for (x, x_dict) in t_dict.items():
                    if np.abs(point_base[1] - x) <= equal_bounds[1]:
                        for (y, y_dict) in x_dict.items():
                            if np.abs(point_base[2] - y) <= equal_bounds[2]:
                                for (z, result_list) in y_dict.items():
                                    if np.abs(point_base[3] - z) <= equal_bounds[3]:
                                        point = (t, x, y, z)
                                        for result in result_list:
                                            filtered_points.append(point)
                                            filtered_results.append(result)
        
        ## filter only one per year
        measurements_filtered = MeasurementsSamePoints()
        
        if only_one_per_year:
            filtered_points = np.array(filtered_points)
            filtered_results = np.array(filtered_results)
            
            years = np.unique(np.floor_divide(filtered_points[:,0], 1))
            point_scale = np.copy(point_base)
            for year in years:
                point_scale[0] = point_base[0]%1 + year
                
                min_index = np.linalg.norm(((filtered_points - point_scale) / equal_bounds), ord=2, axis=1).argmin()
                measurements_filtered.append_value(filtered_points[min_index], filtered_results[min_index])
            number_of_filtered_measurements = len(years)
        else:
            measurements_filtered.append_values(filtered_points, filtered_results)
            number_of_filtered_measurements = len(filtered_results)
        
        logger.debug('{} results for point {} with equal bounds {} filtered.'.format(number_of_filtered_measurements, point, equal_bounds))
        
        return measurements_filtered
    
    
    
    def filter_same_point_except_year(self, point):
        ## prepare point
        point = list(point)
        
        ## filter all measurements with same point
        measurements_filtered = MeasurementsSamePoints()
        
        for (t, t_dict) in  self.value_dict.items():
            if point[0]%1 == t%1:
                point = point.copy()
                point[0] = t
                try:
                    x_dict = t_dict[point[1]]
                    y_dict = x_dict[point[2]]
                    result_list = y_dict[point[3]]
                except KeyError:
                    result_list = None
                
                if result_list is not None:
                    measurements_filtered.extend_value_list(point, result_list)
        
        logger.debug('{} results for point {} filtered.'.format(len(measurements_filtered), point))
        return measurements_filtered
    
    
    def filter_same_points_with_same_function(self, filter_same_point_function, min_values=10):
        assert callable(filter_same_point_function)

        measurements_filtered = MeasurementsSamePoints()
        
        for (point, results) in self.iterator_keys_and_value_lists():
            point = list(point)
            point[0] = point[0] % 1
            
            if not point in measurements_filtered:
                same_point_measurements = filter_same_point_function(point)
                
                transform_function = lambda point, result: (point[0], result)
                same_point_measurements.transform_values(transform_function)
                same_point_value_list = same_point_measurements.values()
                
                if len(same_point_value_list) >= min_values:
                    logger.debug('{} values with point {} passed filter.'.format(len(same_point_value_list), point))
                    measurements_filtered.extend_value_list(point, same_point_value_list)
        
        return measurements_filtered
    
    
    def filter_same_points_except_year(self, min_values=10):
        logger.debug('Filtering results with same indicies with min measurements {}.'.format(min_values))
        
        filter_same_point_function = lambda point: self.filter_same_point_except_year(point)
        return self.filter_same_points_with_same_function(filter_same_point_function, min_values=min_values)
    
    
    def filter_same_points_with_bounds(self, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True, min_values=10):
        logger.debug('Filtering results with same indicies with equal bound {}, discard year {} and min measurements {}.'.format(equal_bounds, discard_year, min_values))
        
        filter_same_point_function = lambda point: self.filter_same_point_with_bounds(point, equal_bounds=equal_bounds, discard_year=discard_year, only_one_per_year=only_one_per_year)
        return self.filter_same_points_with_same_function(filter_same_point_function, min_values=min_values)
    
#     
#     def filter_same_point_except_year(self, point, return_type='measurements_unsorted'):
#         ## prepare point
#         point = list(point)
#         
#         ## filter all measurements with same point
#         filtered_points = []
#         filtered_results = []
#         
#         for (t, t_dict) in  self.value_dict.items():
#             if point[0]%1 == t%1:
#                 point = point.copy()
#                 point[0] = t
#                 try:
#                     x_dict = t_dict[point[1]]
#                     y_dict = x_dict[point[2]]
#                     result_list = y_dict[point[3]]
#                 except KeyError:
#                     result_list = None
#                 
#                 if result_list is not None:
#                     filtered_points.append(point)
#                     filtered_results.append(result_list)
# #                     for result in result_list:
# #                         filtered_points.append(point)
# #                         filtered_results.append(result)
#         
#         logger.debug('{} results for point {} filtered.'.format(len(filtered_results), point))
#         return self._return_items_as_type(filtered_points, filtered_results, return_type=return_type)
#     
#     
#     def filter_same_points_with_same_function(self, filter_same_point_function, min_values=10, return_type='measurements'):
#         assert callable(filter_same_point_function)
#         
#         return_points = []
#         return_value_lists = []
#         
#         for (point, results) in self.iterator_keys_and_value_lists():
#             point = list(point)
#             point[0] = point[0] % 1
#             
#             if not point in return_points:
#                 same_point_measurements = filter_same_point_function(point)
#                 
# #                 transform_function = lambda point, results: [(point[0], result) for result in results]
#                 transform_function = lambda point, result: (point[0], result)
#                 same_point_measurements.transform_values(transform_function)
#                 same_point_value_list = same_point_measurements.values()
#                 
#                 if len(same_point_value_list) >= min_values:
#                     logger.debug('{} values with point {} passed filter.'.format(len(same_point_value_list), point))
#                     return_points.append(point)
#                     return_value_lists.append(same_point_value_list)
#         
#         return self._return_items_as_type(return_points, return_value_lists, return_type=return_type)
#     
#     
#     def filter_same_points_except_year(self, min_values=10, return_type='measurements'):
#         logger.debug('Filtering results with same indicies with min measurements {}.'.format(min_values))
#         
#         filter_same_point_function = lambda point: self.filter_same_point_except_year(point, return_type='measurements_unsorted')
#         return self.filter_same_points_with_same_function(filter_same_point_function, min_values=min_values, return_type=return_type)
#     
#     
#     def filter_same_points_with_bounds(self, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True, min_values=10, return_type='measurements'):
#         logger.debug('Filtering results with same indicies with equal bound {}, discard year {} and min measurements {}.'.format(equal_bounds, discard_year, min_values))
#         
#         filter_same_point_function = lambda point: self.filter_same_point_with_bounds(point, equal_bounds=equal_bounds, discard_year=discard_year, only_one_per_year=only_one_per_year)
#         return self.filter_same_points_with_same_function(filter_same_point_function, min_values=min_values, return_type=return_type)
    
    
    
    
    
    # def correlation(self, min_values=10, stationary=False, return_type='array'):
    #     logger.debug('Calculate correlation of values with at least {} values.'.format(min_values))
    #     
    #     keys = []
    #     value_lists = []
    #     
    #     ## iterate over each pair of measurement indices
    #     index_of_measurement_1 = 0
    #     for (key_1, transformed_value_list_1) in self.iterator_keys_and_value_lists():
    #         index_of_measurement_2 = 0
    #         for (key_2, transformed_value_list_2) in self.iterator_keys_and_value_lists():
    #             
    #             ## skip previous values
    #             if index_of_measurement_2 > index_of_measurement_1:
    #                 
    #                 ## get matching values
    #                 res = []
    #                 for (t1, r1) in transformed_value_list_1:
    #                     for (t2, r2) in transformed_value_list_2:
    #                         t_diff = np.abs((t1 - (key_1[0]%1)) - (t2 - (key_2[0]%1)))
    #                         if t_diff < 0.9: # could also be 1.0, but 0.9 for rounding errors
    #                             res.append([r1, r2])
    #                 res = np.array(res)
    #                 n = len(res)
    #                 
    #                 ## calculate correlation
    #                 if n >= min_values:
    #                     x1 = res[:,0]
    #                     x2 = res[:,1]
    #                     
    #                     m1 = x1.mean()
    #                     m2 = x2.mean()
    #                     s1 = np.sqrt(np.sum((x1 - m1)**2))
    #                     s2 = np.sqrt(np.sum((x2 - m2)**2))
    #                     
    #                     if s1 > 0 and s2 > 0:
    #                         correlation = np.sum((x1 - m1) / s1 * (x2 - m2) / s2)
    #                         
    #                         if stationary:
    #                             correlation_key = np.abs(np.asarray(key_1) - np.asarray(key_2))
    #                             correlation_key[1] = min(correlation_key[1], 360-correlation_key[1])
    #                         else:
    #                             both_keys = np.array((key_1, key_2))
    #                             both_keys = both_keys[np.lexsort(both_keys.T)]
    #                             correlation_key = tuple(both_keys.flat)
    #                         
    #                         keys.append(correlation_key)
    #                         value_lists.append([(n, correlation)])
    #                         
    #                         logger.debug('Correlation {} calculated with {} values for index {}.'.format(correlation, n, correlation_key))
    #                     else:
    #                         logger.warning('Correlation for key {} and {} could not be calculated since a sample standard deviation is zero. Skippping this correlation'.format(key_1, key_2))
    #                         
    #             
    #             index_of_measurement_2 += 1
    #         index_of_measurement_1 += 1
    #     
    #     
    #     return self._return_items_as_type(keys, value_lists, return_type=return_type)
    
    
    
    
    ## total correlogram and correlation (autocorrelation)
    
    def _get_first_dim_shifted(self, measurements_dict_list, shift, same_bound, wrap_around_range=None):
        logger.debug('Getting first dim shifted with shift %f and same bound %f.' % (shift, same_bound))
        
        if self.sorted:
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
            
            return shift_list
            
        
        else:
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
        n = 0
        for (result_list, result_shifted_list) in shift_list:
            n += len(result_list) * len(result_shifted_list)
        
        ## set values
        array = np.empty((n, 2))
        i = 0
        for (result_list, result_shifted_list) in shift_list:
            for result in result_list:
                for result_shifted in result_shifted_list:
                    array[i, 0] = result
                    array[i, 1] = result_shifted
                    i += 1
        
        assert i == n
        
        return array
    
    
    
    def _calculate_total_correlation_from_shift_list(self, shift_list, is_normalized=False):
        if not is_normalized:
            #TODO mean and sd for each result list
            shift_array = self._get_array_from_shift_list(shift_list)
            number = shift_array.shape[0]
            
            
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
            (value, number_of_measurements) = calculation_function(measurements_dict_list[dim])
            if number_of_measurements >= minimum_measurements:
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



class MeasurementsSamePoints(Measurements):    
    
    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)
    
    
    # @staticmethod
    # def point_pair_to_key(points, cross_year=False, stationary=False):  
    #     assert len(points) == 2
    #     
    #     if stationary:
    #         key = np.abs(np.asarray(points[0]) - np.asarray(points[1]))
    #         key[1] = min(key[1], 360-key[1])
    #         key = tuple(key)
    #     else:
    #         both_keys = np.array((points[0], points[1]))
    #         both_keys = both_keys[np.lexsort(both_keys.T)]
    #         key = tuple(both_keys.flat)
    #     
    #     return key
    
    
    
    # ## compute values
    # 
    # # def correlation_or_covariance(self, value_type, min_values=10, stationary=False, return_type='array'):
    # #     logger.debug('Calculate {} with at least {} values.'.format(value_type, min_values))
    # #     
    # #     POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    # #     if value_type not in POSSIBLE_VALUE_TYPES:
    # #         raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))
    # #     else:
    # #         calculate_correlation = value_type == POSSIBLE_VALUE_TYPES[0]
    # #     
    # #     keys = []
    # #     value_lists = []
    # #     
    # #     ## iterate over each pair of measurement indices
    # #     index_of_measurement_1 = 0
    # #     for (key_1, transformed_value_list_1) in self.iterator_keys_and_value_lists():
    # #         index_of_measurement_2 = 0
    # #         for (key_2, transformed_value_list_2) in self.iterator_keys_and_value_lists():
    # #             
    # #             ## skip previous values
    # #             if index_of_measurement_2 > index_of_measurement_1:
    # #                 
    # #                 ## get matching values
    # #                 matching_results = []
    # #                 for (t1, r1) in transformed_value_list_1:
    # #                     for (t2, r2) in transformed_value_list_2:
    # #                         t_diff = np.abs((t1 - (key_1[0]%1)) - (t2 - (key_2[0]%1)))
    # #                         if t_diff < 0.9: # could also be 1.0, but 0.9 for rounding errors
    # #                             matching_results.append([r1, r2])
    # #                 matching_results = np.array(matching_results)
    # #                 n = len(matching_results)
    # #                 
    # #                 ## calculate correlation
    # #                 if n >= min_values:
    # #                     x1 = matching_results[:,0]
    # #                     x2 = matching_results[:,1]
    # #                     
    # #                     m1 = x1.mean()
    # #                     m2 = x2.mean()
    # #                     
    # #                     if calculate_correlation:
    # #                         s1 = np.sqrt(np.sum((x1 - m1)**2))
    # #                         s2 = np.sqrt(np.sum((x2 - m2)**2))
    # #                     
    # #                     if not calculate_correlation or (s1 > 0 and s2 > 0):
    # #                         if calculate_correlation:
    # #                             value = np.sum(((x1 - m1) / s1) * ((x2 - m2) / s2))
    # #                         else:
    # #                             value = np.sum((x1 - m1) * (x2 - m2))
    # #                         
    # #                         # if stationary:
    # #                         #     value_key = np.abs(np.asarray(key_1) - np.asarray(key_2))
    # #                         #     value_key[1] = min(value_key[1], 360-value_key[1])
    # #                         # else:
    # #                         #     both_keys = np.array((key_1, key_2))
    # #                         #     both_keys = both_keys[np.lexsort(both_keys.T)]
    # #                         #     value_key = tuple(both_keys.flat)
    # #                         value_key = self.key_for_point_pair((key_1, key_2), stationary)
    # #                         
    # #                         keys.append(value_key)
    # #                         value_lists.append([(n, value)])
    # #                         
    # #                         logger.debug('{} {} calculated with {} values for index {}.'.format(value_type, value, n, value_key))
    # #                     else:
    # #                         logger.warning('Correlation for key {} and {} could not be calculated since a sample standard deviation is zero. Skippping this correlation'.format(key_1, key_2))
    # #                         
    # #             
    # #             index_of_measurement_2 += 1
    # #         index_of_measurement_1 += 1
    # #     
    # #     return self._return_items_as_type(keys, value_lists, return_type=return_type)
    # 
    # 
    # def correlation_or_covariance(self, value_type, min_values=10, stationary=False, return_type='array'):
    #     logger.debug('Calculate {} with at least {} values.'.format(value_type, min_values))
    #     
    #     POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    #     if value_type not in POSSIBLE_VALUE_TYPES:
    #         raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))
    #     else:
    #         calculate_correlation = value_type == POSSIBLE_VALUE_TYPES[0]
    #     
    #     keys = []
    #     value_lists = []
    #     
    #     ## iterate over each pair of measurement indices
    #     index_of_measurement_1 = 0
    #     for (key_1, transformed_value_list_1) in self.iterator_keys_and_value_lists():
    #         index_of_measurement_2 = 0
    #         for (key_2, transformed_value_list_2) in self.iterator_keys_and_value_lists():
    #             
    #             ## skip previous values
    #             if index_of_measurement_2 > index_of_measurement_1:
    #                 
    #                 ## calculate for both t_diffs
    #                 t_diff = np.abs(key_1[0] % 1 - key_2[0] % 1)
    #                 
    #                 if np.isclose(t_diff, 0):
    #                     cross_year_values = (False,)
    #                 else:
    #                     cross_year_values = (False, True)
    #                 
    #                 for cross_year in cross_year_values:
    #                     ## apply cross year
    #                     if cross_year:
    #                         t_diff = 1 - t_diff
    #                     
    #                     ## get matching values
    #                     matching_results = []
    #                     for (t1, r1) in transformed_value_list_1:
    #                         for (t2, r2) in transformed_value_list_2:
    #                             if np.isclose(np.abs(t1 - t2), t_diff):
    #                                 matching_results.append([r1, r2])
    #                     matching_results = np.array(matching_results)
    #                     n = len(matching_results)
    #                     
    #                     ## calculate correlation
    #                     if n >= min_values:
    #                         x1 = matching_results[:,0]
    #                         x2 = matching_results[:,1]
    #                         
    #                         m1 = x1.mean()
    #                         m2 = x2.mean()
    #                         
    #                         if calculate_correlation:
    #                             s1 = np.sqrt(np.sum((x1 - m1)**2))
    #                             s2 = np.sqrt(np.sum((x2 - m2)**2))
    #                         
    #                         if not calculate_correlation or (s1 > 0 and s2 > 0):
    #                             if calculate_correlation:
    #                                 value = np.sum(((x1 - m1) / s1) * ((x2 - m2) / s2))
    #                             else:
    #                                 value = np.sum((x1 - m1) * (x2 - m2))
    #                             
    #                             point_1 = key_1                                
    #                             point_2 = key_2
    #                             if cross_year:
    #                                 if point_1[0] <= point_2[0]:
    #                                     point_1 = list(point_1)
    #                                     point_1[0] = point_1[0] + 1
    #                                 else:
    #                                     point_2 = list(point_2)
    #                                     point_2[0] = point_2[0] + 1
    #                             
    #                             value_key = self.key_for_point_pair((point_1, point_2), stationary)
    #                             
    #                             keys.append(value_key)
    #                             value_lists.append([(n, value)])
    #                             
    #                             logger.debug('{} {} calculated with {} values for index {}.'.format(value_type, value, n, value_key))
    #                         else:
    #                             logger.warning('Correlation for key {} and {} could not be calculated since a sample standard deviation is zero. Skippping this correlation'.format(key_1, key_2))
    #                         
    #             
    #             index_of_measurement_2 += 1
    #         index_of_measurement_1 += 1
    #     
    #     return self._return_items_as_type(keys, value_lists, return_type=return_type)
    
    
    # ## compute values
    # 
    # def correlation_or_covariance(self, value_type, min_values=10, stationary=False):
    #     logger.debug('Calculate {} with at least {} values.'.format(value_type, min_values))
    #     
    #     ## check value type
    #     POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    #     if value_type not in POSSIBLE_VALUE_TYPES:
    #         raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))
    #     else:
    #         calculate_correlation = value_type == POSSIBLE_VALUE_TYPES[0]
    #     
    #     ## prepare value measurement dict
    #     if stationary:
    #         value_measurements = MeasurementsCovarianceStationary()
    #     else:
    #         value_measurements = MeasurementsCovariance()
    #     
    #     
    #     ## iterate over each pair of measurement indices
    #     index_of_measurement_1 = 0
    #     for (key_1, transformed_value_list_1) in self.iterator_keys_and_value_lists():
    #         index_of_measurement_2 = 0
    #         for (key_2, transformed_value_list_2) in self.iterator_keys_and_value_lists():
    #             
    #             ## skip previous values
    #             if index_of_measurement_2 > index_of_measurement_1:
    #                 
    #                 ## calculate for both t_diffs
    #                 t_diff = np.abs(key_1[0] % 1 - key_2[0] % 1)
    #                 
    #                 if np.isclose(t_diff, 0):
    #                     cross_year_values = (False,)
    #                 else:
    #                     cross_year_values = (False, True)
    #                 
    #                 for cross_year in cross_year_values:
    #                     ## apply cross year
    #                     if cross_year:
    #                         t_diff = 1 - t_diff
    #                     
    #                     ## get matching values
    #                     matching_results = []
    #                     for (t1, r1) in transformed_value_list_1:
    #                         for (t2, r2) in transformed_value_list_2:
    #                             if np.isclose(np.abs(t1 - t2), t_diff):
    #                                 matching_results.append([r1, r2])
    #                     matching_results = np.array(matching_results)
    #                     n = len(matching_results)
    #                     
    #                     ## calculate correlation
    #                     if n >= min_values:
    #                         x1 = matching_results[:,0]
    #                         x2 = matching_results[:,1]
    #                         
    #                         m1 = x1.mean()
    #                         m2 = x2.mean()
    #                         
    #                         if calculate_correlation:
    #                             s1 = np.sqrt(np.sum((x1 - m1)**2))
    #                             s2 = np.sqrt(np.sum((x2 - m2)**2))
    #                         
    #                         if not calculate_correlation or (s1 > 0 and s2 > 0):
    #                             ## calculate value
    #                             if calculate_correlation:
    #                                 value = np.sum(((x1 - m1) / s1) * ((x2 - m2) / s2))
    #                             else:
    #                                 value = np.sum((x1 - m1) * (x2 - m2))                                
    #                             value = [(n, value)]
    #                             
    #                             ## prepare key pair
    #                             keys = [list(key_1), list(key_2)]
    #                             if cross_year:
    #                                 if keys[0][0] <= keys[1][0]:
    #                                     keys[0][0] = keys[0][0] + 1
    #                                 else:
    #                                     keys[1][0] = keys[1][0] + 1
    #                             
    #                             ## add value to value dict
    #                             value_measurements.append_value(keys, value)
    #                             
    #                             logger.debug('{} {} calculated with {} values for index {}.'.format(value_type, value, n, keys))
    #                         else:
    #                             logger.warning('Correlation for key {} and {} could not be calculated since a sample standard deviation is zero. Skipping this correlation'.format(keys[0], keys[1]))
    #                         
    #             
    #             index_of_measurement_2 += 1
    #         index_of_measurement_1 += 1
    #     
    #     return value_measurements
    
    # 
    # ## compute values
    # 
    # def correlation_or_covariance(self, value_type, min_values=10, stationary=False, max_year_diff=1):
    #     logger.debug('Calculate {} with at least {} values.'.format(value_type, min_values))
    #     
    #     ## check value type
    #     POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    #     if value_type not in POSSIBLE_VALUE_TYPES:
    #         raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))
    #     else:
    #         calculate_correlation = value_type == POSSIBLE_VALUE_TYPES[0]
    #     
    #     ## prepare value measurement dict
    #     if stationary:
    #         value_measurements = MeasurementsCovarianceStationary()
    #     else:
    #         value_measurements = MeasurementsCovariance()
    #     
    #     
    #     ## iterate over each pair of measurement indices
    #     index_of_measurement = [0, 0]
    #     # index_of_measurement_1 = 0
    #     for (key_1, transformed_value_list_1) in self.iterator_keys_and_value_lists():
    #         index_of_measurement[1] = 0
    #         # index_of_measurement_2 = 0
    #         for (key_2, transformed_value_list_2) in self.iterator_keys_and_value_lists():
    #             
    #             ## skip previous values
    #             if index_of_measurement_2 > index_of_measurement_1 or max_year_diff > 1:
    #                 
    #                 ## calculate if cross year
    #                 if np.isclose(key_1[0] % 1, key_2[0] % 1):
    #                     cross_year_values = (False,)
    #                 else:
    #                     cross_year_values = (False, True)
    #                 
    #                 for desired_year_diff in range(max_year_diff):
    #                     for cross_year in cross_year_values:
    #                         ## calculate desired year diff and t fraction diff
    #                         desired_t_fraction_diff = np.abs(key_1[0] % 1 - key_2[0] % 1)
    #                         
    #                         if cross_year:
    #                             desired_t_fraction_diff = 1 - desired_t_fraction_diff
    #                             desired_year_diff += 1
    #                         
    #                         ## get matching values
    #                         matching_results = []
    #                         for (t1, r1) in transformed_value_list_1:
    #                             for (t2, r2) in transformed_value_list_2:
    #                                 year_diff = np.abs(int(t1) -  int(t2))
    #                                 if year_diff == desired_year_diff:
    #                                     t_fraction_diff = np.abs(t1 % 1 - t2 % 1)
    #                                     if np.isclose(t_fraction_diff, desired_t_fraction_diff):
    #                                         matching_results.append([r1, r2])
    #                         matching_results = np.array(matching_results)
    #                         n = len(matching_results)
    #                         
    #                         ## calculate correlation
    #                         if n >= min_values:
    #                             x1 = matching_results[:,0]
    #                             x2 = matching_results[:,1]
    #                             
    #                             m1 = x1.mean()
    #                             m2 = x2.mean()
    #                             
    #                             if calculate_correlation:
    #                                 s1 = np.sqrt(np.sum((x1 - m1)**2))
    #                                 s2 = np.sqrt(np.sum((x2 - m2)**2))
    #                             
    #                             if not calculate_correlation or (s1 > 0 and s2 > 0):
    #                                 ## calculate value
    #                                 if calculate_correlation:
    #                                     value = np.sum(((x1 - m1) / s1) * ((x2 - m2) / s2))
    #                                 else:
    #                                     value = np.sum((x1 - m1) * (x2 - m2))                                
    #                                 value = (n, value)
    #                                 
    #                                 ## prepare key pair
    #                                 keys = [list(key_1), list(key_2)]
    #                                 if keys[0][0] <= keys[1][0]:
    #                                     lower_t_key_index = 0
    #                                 else:
    #                                     lower_t_key_index = 1
    #                                 if cross_year:
    #                                     keys[lower_t_key_index][0] = keys[lower_t_key_index][0] + desired_year_diff
    #                                 else:
    #                                     keys[1-lower_t_key_index][0] = keys[1-lower_t_key_index][0] + desired_year_diff
    #                                 # if cross_year:
    #                                 #     if keys[0][0] <= keys[1][0]:
    #                                 #         keys[0][0] = keys[0][0] + desired_year_diff
    #                                 #     else:
    #                                 #         keys[1][0] = keys[1][0] + desired_year_diff
    #                                 
    #                                 ## add value to value dict
    #                                 value_measurements.append_value(keys, value)
    #                                 
    #                                 logger.debug('{} {} calculated with {} values for index {}.'.format(value_type, value, n, keys))
    #                             else:
    #                                 logger.warning('Correlation for key {} and {} could not be calculated since a sample standard deviation is zero. Skipping this correlation'.format(keys[0], keys[1]))
    #             
    #             
    #             index_of_measurement_2 += 1
    #         index_of_measurement_1 += 1
    #     
    #     return value_measurements
    
    
    
    ## compute values
    
    def correlation_or_covariance(self, value_type, min_values=10, stationary=False, max_year_diff=float('inf')):
        logger.debug('Calculate {} with at least {} values, stationary {} and max_year_diff {}.'.format(value_type, min_values, stationary, max_year_diff))
        
        ## check value type
        POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
        if value_type not in POSSIBLE_VALUE_TYPES:
            raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))
        else:
            calculate_correlation = value_type == POSSIBLE_VALUE_TYPES[0]
        
        ## check max_year_diff
        if max_year_diff is None or max_year_diff == float('inf'):
            t = self.values()[:,0]
            max_year_diff = int(np.ceil(t.max() - t.min()))
            logger.debug('Using max_year_diff {}.'.format(max_year_diff))
        
        ## prepare value measurement dict
        if stationary:
            value_measurements = MeasurementsCovarianceStationary()
        else:
            value_measurements = MeasurementsCovariance()
        
        
        ## iterate over each pair of measurement indices
        index_of_measurement = [0, 0]
        for (key_0, transformed_value_list_0) in self.iterator_keys_and_value_lists():
            index_of_measurement[1] = 0
            for (key_1, transformed_value_list_1) in self.iterator_keys_and_value_lists():
                
                ## skip previous values
                if index_of_measurement[1] > index_of_measurement[0] or (index_of_measurement[1] == index_of_measurement[0] and max_year_diff > 1):
                    
                    ## make keys to list
                    keys = (tuple(key_0), tuple(key_1))
                    
                    ## calculate all desired year offsets
                    desired_year_offsets = tuple(range(1, max_year_diff))
                    if not np.allclose(keys[0], keys[1]):
                        desired_year_offsets += tuple(range(-max_year_diff+1, 1))
                    
                    if not np.isclose(keys[0][0], keys[1][0]):
                        if keys[0][0] < keys[1][0]:
                            desired_year_offsets +=  (max_year_diff,)
                        else:
                            desired_year_offsets +=  (-max_year_diff,)
                    
                    ## for all year offsets
                    for desired_year_offset in desired_year_offsets:
                        
                        ## get values with desired year offset and t fraction diff
                        matching_results = []
                        for (t0, r0) in transformed_value_list_0:
                            for (t1, r1) in transformed_value_list_1:
                                year_offset = int(t0) - int(t1)
                                if year_offset == desired_year_offset:
                                    matching_results.append([r0, r1])
                        matching_results = np.array(matching_results)
                        n = len(matching_results)
                        
                        
                        ## if enough measurements
                        if n >= min_values:      
                                                
                            ## calculate auxiliary values
                            x0 = matching_results[:,0]
                            x1 = matching_results[:,1]
                            
                            m0 = x0.mean()
                            m1 = x1.mean()
                            
                            if calculate_correlation:
                                s0 = np.sqrt(np.sum((x0 - m0)**2))
                                s1 = np.sqrt(np.sum((x1 - m1)**2))
                            
                            if not calculate_correlation or (s0 > 0 and s1 > 0):
                                ## calculate value
                                if calculate_correlation:
                                    value = np.sum(((x0 - m0) / s0) * ((x1 - m1) / s1))
                                    assert value >= -1 and value <= 1
                                else:
                                    value = np.sum((x0 - m0) * (x1 - m1))
                                
                                value = (n, value)
                                
                                ## prepare key pair
                                value_keys = [list(keys[0]), list(keys[1])]
                                if desired_year_offset >= 0:
                                    value_keys[0][0] = value_keys[0][0] + desired_year_offset
                                else:
                                    value_keys[1][0] = value_keys[1][0] - desired_year_offset
                                
                                ## add value to value dict
                                value_measurements.append_value(value_keys, value)
                                
                                logger.debug('{} {} calculated with {} values for index {}.'.format(value_type, value, n, value_keys))
                            else:
                                logger.warning('Correlation for key {} and {} could not be calculated since a sample standard deviation is zero. Skipping this correlation'.format(keys[0], keys[1]))
                
                
                index_of_measurement[1] += 1
            index_of_measurement[0] += 1
        
        return value_measurements
    
    
    def correlation(self, min_values=10, stationary=False):
        return self.correlation_or_covariance('correlation', min_values=min_values, stationary=stationary)    
    
    
    def covariance(self, min_values=10, stationary=False):
        return self.correlation_or_covariance('covariance', min_values=min_values, stationary=stationary)




class MeasurementsCovariance(util.multi_dict.MultiDictPermutablePointPairs):
    
    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)
        self._year_len = 1
    
    
    def _preprocess_keys(self, keys):
        ## copy keys
        keys = list(keys)
        keys[0] = list(keys[0])
        keys[1] = list(keys[1])
        
        ## remove lower year offset
        year_len = self.year_len
        lower_years = min([int(keys[0][0]/year_len), int(keys[1][0]/year_len)])
        for key in keys:
            key[0] = key[0] - lower_years * year_len
        
        ## get value
        return super()._preprocess_keys(keys)
    
    
    @property
    def year_len(self):
        return self._year_len
    
    
    ## transform keys
    
    def coordinates_to_map_indices(self, lsm):
        logger.debug('Transforming in {} coordinates to map indices of {}'.format(self, lsm))
        
        self._year_len = lsm.t_dim
        self.transform_keys(lambda keys: (lsm.coordinate_to_map_index(*keys[0], discard_year=False), lsm.coordinate_to_map_index(*keys[1], discard_year=False)))
    
    
    def map_indices_to_coordinates(self, lsm):
        logger.debug('Transforming in {} map indices of {} to coordinates'.format(self, lsm))
        
        self._year_len = 1
        self.transform_keys(lambda indices: (lsm.map_index_to_coordinate(*indices[0]), lsm.map_index_to_coordinate(*indices[1])))
    
    
    
    
    ## io
    def save(self, file):
        only_dict = self._year_len != 1
        super().save(file, only_dict=only_dict)
    #     logger.debug('Saving {} to {}.'.format(self, file))
    #     util.io.object.save(file, self)
    #  
    # 
    # @classmethod
    # def load(cls, file):
    #     logger.debug('Loading {} from {}.'.format(cls.__name__, file))
    #     
    #     obj = util.io.object.load(file)
    #     return obj


# class MeasurementsCovariance(util.multi_dict.MultiDictPermutablePointPairs):
#     
#     def __init__(self, sorted=False):
#         super().__init__(sorted=sorted)
#         self.year_len = 1
#     
#     
#     def _preprocess_keys(self, keys):
#         t_diff = np.abs(keys[0][0] - keys[1][0])
#         year_len = self.year_len
#         
#         if t_diff < year_len:
#             ## copy keys
#             keys = list(keys)
#             keys[0] = list(keys[0])
#             keys[1] = list(keys[1])
#             
#             ## check if cross year
#             cross_year = int(keys[0][0]/year_len) != int(keys[1][0]/year_len)
#             
#             ## discard year
#             for key in keys:
#                 key[0] = key[0] % year_len
#             
#             ## prepare keys if cross year
#             if cross_year:
#                 if keys[0][0] <= keys[1][0]:
#                     keys[0][0] = keys[0][0] + year_len
#                 else:
#                     keys[1][0] = keys[1][0] + year_len
#             
#             return super(MeasurementsCovariance,MeasurementsCovariance)._preprocess_keys(keys)
#         else:
#             raise KeyError('Keys {} not allowed. Only keys with time diff {} less then year len {}!'.format(keys, t_diff, year_len))
#     
#     
#     def coordinates_to_map_indices(self, lsm):
#         self.year_len = lsm.t_dim
#         self.transform_keys(lambda keys: (lsm.coordinate_to_map_index(*keys[0]), lsm.coordinate_to_map_index(*keys[1], discard_year=False)))
#     
#     def map_indices_to_coordinates(self, lsm):
#         self.year_len = 1
#         self.transform_keys(lambda indices: (lsm.map_index_to_coordinate(*indices[0]), lsm.map_index_to_coordinate(*indices[1])))
    
    # @staticmethod
    # def categorized_point_pair_without_year_to_key(points, cross_year, stationary=False):  
    #     assert len(points) == 2
    #     points = list(points)

    #       if cross_year:
    #         if points[0][0] <= points[1][0]:
    #             points[0] = list(points[0])
    #             points[0][0] = points[0][0] + 1
    #         else:
    #             points[1] = list(points[1])
    #             points[1][0] = points[1][0] + 1
    #     
    #     if stationary:
    #         key = np.abs(np.asarray(points[0]) - np.asarray(points[1]))
    #         key[1] = min(key[1], 360-key[1])
    #         key = tuple(key)
    #     else:
    #         both_keys = np.array((points[0], points[1]))
    #         both_keys = both_keys[np.lexsort(both_keys.T)]
    #         key = tuple(both_keys.flat)
    #     
    #     return key



class MeasurementsCovarianceStationary(util.multi_dict.MultiDictDiffPointPairs, Measurements):
    
    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)


    

# class MeasurementsUnsorted(Measurements):
#     
#     def __init__(self):
#         super(Measurements, self).__init__(sorted=False)
#     
#     # def new_like(self, sorted=None):
#     #     if sorted is None:
#     #         sorted = self.sorted
#     #     new = Measurements(sorted=sorted)
#     #     return new
# 
# 
# 
# class MeasurementsSorted(Measurements):
#     
#     def __init__(self):
#         super(Measurements, self).__init__(sorted=True)
#     
#     # def new_like(self, sorted=None):
#     #     if sorted is None:
#     #         sorted = self.sorted
#     #     new = Measurements(sorted=sorted)
#     #     return new



# class MeasurementsUnsorted():
#     
#     def __init__(self):
#         self.measurements_dict = dict()
#         self.SUPPORTED_RETURN_TYPES = ('array', 'self', 'self_type', 'measurements', 'measurements_unsorted', 'measurements_sorted')
#     
#     
#     ## mapping methods
#     
#     def get_result_list(self, point):
#         value = self.measurements_dict
#         
#         for i in point:
#             value = value[i]
#         
#         return value
#     
#     def __getitem__(self, key):
#         return self.get_result_list(key)
#     
#     
#     def set_result_list(self, point, result_list):
#         result_list = list(result_list)
#         
#         dictionary = self.measurements_dict
#         measurements_dict_type = type(dictionary)
#         
#         n = len(point)
#         for i in range(n-1):
#             dictionary = dictionary.setdefault(point[i], measurements_dict_type())
#         dictionary[point[n-1]] = result_list
#     
#     def __setitem__(self, key, value):
#         self.set_result_list(key, value)
#     
#     
#     def has_result_list(self, point):
#         value = self.measurements_dict
#         
#         for i in point:
#             try:
#                 value = value[i]
#             except KeyError:
#                 return False
#         
#         return len(value) > 0
#     
#     def __contains__(self, key):
#         return self.has_result_list(key)
#     
#     @property
#     def len(self):
#         total_len = 0
#         for (point, result_list) in self.iterator_keys_and_value_lists():
#             total_len += len(result_list) 
#         
#         return total_len
#     
#     def __len__(self):
#         return self.len
#     
#     
#     ## add & remove
#     
#     def _get_or_init_result_list(self, point):
#         dictionary = self.measurements_dict
#         measurements_dict_type = type(dictionary)
#         
#         n = len(point)
#         for i in range(n-1):
#             dictionary = dictionary.setdefault(point[i], measurements_dict_type())
#         result_list = dictionary.setdefault(point[n-1], [])
#         
#         return result_list
#     
#     def extend_result_list(self, point, result_list):
#         self._get_or_init_result_list(point).extend(result_list)
#     
#     def append_result_list(self, point, result):
#         self._get_or_init_result_list(point).append(result)
#     
#     
#     def _add_result_lists(self, points, result_lists, add_function):
#         assert callable add_function
#         
#         if len(points) != len(result_lists):
#             raise ValueError('Len of points {} and len of results {} have to be the same!'.format(len(points), len(result_lists)))
#         
#         results_len = len(result_lists)
#         logger.debug('Adding {} measurements.'.format(results_len))
#         for i in range(result_lists):
#             add_function(points[i], result_lists[i])
#     
#     def extend_result_lists(self, points, result_lists):
#         add_function = lambda point, result_list: self.extend_result_list(point, result_list)
#         self._add_result_lists(points, result_lists, add_function)
#     
#     def append_result_lists(self, points, result_lists):
#         add_function = lambda point, result_list: self.append_result_list(point, result_list)
#         self._add_result_lists(points, result_lists, add_function)
#     
# #     def add_result(self, point, result):
# #         dictionary = self.measurements_dict
# #         measurements_dict_type = type(dictionary)
# #         
# #         n = len(point)
# #         for i in range(n-1):
# #             dictionary = dictionary.setdefault(point[i], measurements_dict_type())
# #         result_list = dictionary.setdefault(point[n-1], [])
# #         try:
# #             result_list.extend(result)
# #         except TypeError:
# #             result_list.append(result)
#     
#     
#     def add_result(self, point, result):
#         self.append_result_list(point, result)
#     
#     
#     def add_results(self, points, results):
#         if len(points) != len(results):
#             raise ValueError('Len of points {} and len of results {} have to be the same!'.format(len(points), len(results)))
#         
#         results_len = len(results)
#         logger.debug('Adding {} measurements.'.format(results_len))
#         for i in range(results_len):
#             self.extend_result_list(points[i], results[i])
#     
#     
#     def clear(self):
#         self.measurements_dict = type(self.measurements_dict)()
#     
#     
#     
#     ## access
#     
#     def all_points(self):
#         all_points = []
#         for (point, results) in self.iterator_keys_and_value_lists():
#             for result in results:
#                 all_points.append(point)
#         
#         all_points = np.array(all_points)
#         assert all_points.ndim == 2
#         assert all_points.shape[1] == 4
#         return all_points
#     
#     
#     def all_results(self):
#         all_results = []
#         for (point, results) in self.iterator_keys_and_value_lists():
#             all_results.extend(results)
#         
#         all_results = np.array(all_results)
#         return all_results
#     
#     
#     def all_points_and_results(self):
#         all_points = self.all_points()
#         all_results = self.all_results()
#         if all_results.ndim == 1:
#             all_results = all_results[:, np.newaxis]
#         return np.concatenate([all_points, all_results], axis=1)
#     
#     ## io
#     
#     def save(self, file):
#         logger.debug('Saving measurements at %s.', file)
#         util.io.io.save_object(self.measurements_dict, file)
#     
#     
#     def load(self, file):
#         logger.debug('Loading measurements from %s.', file)
#         self.measurements_dict = util.io.io.load_object(file)
#         return self
#     
#     
#     
#     ## iterate
#     
#     def iterator_keys_and_value_lists(self):
#         measurements_dict = self.measurements_dict
#         yield from self._iterate_generator_measurements_dict(measurements_dict)
# #         for (t, t_dict) in measurements_dict.items():
# #             for (x, x_dict) in t_dict.items():
# #                 for (y, y_dict) in x_dict.items():
# #                     for (z, results_list) in y_dict.items():
# #                         yield ([t, x, y, z], results_list)
#     
#     def _iterate_generator_measurements_dict(self, measurements_dict):
#         for (t, t_dict) in measurements_dict.items():
#             for (x, x_dict) in t_dict.items():
#                 for (y, y_dict) in x_dict.items():
#                     for (z, results_list) in y_dict.items():
#                         yield ([t, x, y, z], results_list)
#         
#     
#     
#     
# #     def iterate(self, fun, minimum_measurements=1, return_type='array'):
# # #         measurements_dict = self.measurements_dict
# #         
# #         ## check input
# #         if return_type not in ('array', 'self', 'measurements'):
# #             raise ValueError('Unknown return_type "%s". Only "array", "self" and "measurements" are supported.' % return_type)
# #         
# #         ## init
# #         if return_type is 'array':
# #             values = []
# #         else:
# #             values = type(self)()
# #         
# #         ## iterate
# #         for (t, x, y, z, results_list) in self.iterator_keys_and_value_lists():
# #             if len(results_list) >= minimum_measurements:
# #                 results = np.array(results_list)
# #                 value = fun(results)
# #                 
# #                 ## insert
# #                 if return_type is 'array':
# #                     row = [t, x, y, z, value]
# #                     values.append(row)
# #                 else:
# #                     index = (t, x, y, z)
# #                     values.add_result(index, value)
# #         
# #         ## finishing
# #         if return_type is 'array':
# #             values = np.array(values)
# #             logger.debug('{} values calculated.'.format(values.shape[0]))
# #         
# #         return values
#     
#     
# #     def iterate(self, fun, minimum_measurements=1, return_type='array'):
# #         ## init
# #         points = []
# #         values = []
# #         
# #         ## iterate
# #         for (point, results) in self.iterator_keys_and_value_lists():
# #             if len(results) >= minimum_measurements:
# #                 results = np.asarray(results)
# #                 value = fun(results)
# #                 
# #                 ## insert
# #                 points.append(point)
# #                 values.append(value)
# #         
# #         ## finishing
# #         return self._return_points_and_values_as_type(points, values, return_type=return_type)
#     
#     
#     def iterate_points_and_results(self, fun, min_measurements=1, return_type='array'):
#         assert callable(fun)
#         
#         ## init
#         points = []
#         values = []
#         
#         ## iterate
#         for (point, results) in self.iterator_keys_and_value_lists():
#             if len(results) >= min_measurements:
#                 point = np.asarray(point)
#                 results = np.asarray(results)
#                 value = fun(point, results)
#                 
#                 ## insert
#                 points.append(point)
#                 values.append(value)
#         
#         ## finishing
#         return self._return_points_and_values_as_type(points, values, return_type=return_type)
#     
#     
#     def iterate_results(self, fun, min_measurements=1, return_type='array'):
#         fun_wrapper = lambda point, results: fun(results)
#         return self.iterate_points_and_results(fun_wrapper, min_measurements=min_measurements, return_type=return_type)
#     
#     
#     def _return_points_and_values_as_type(self, points, values, return_type=None):
#         ## chose default
#         if return_type is None:
#             return_type = 'array'
#         if return_type == 'measurements':
#             return_type = 'measurements_unsorted'
#         
#         ## check input
#         if return_type not in self.SUPPORTED_RETURN_TYPES:
#             raise ValueError('Unknown return_type "{}". Only {} are supported.'.format(return_type, self.SUPPORTED_RETURN_TYPES))
#         
#         if len(points) != len(values):
#             raise ValueError('Len of points {} and len of values {} have to be the same!'.format(len(points), len(values)))
#         
#         
#         ## return of measurements type
#         if return_type in ('self', 'self_type', 'measurements_unsorted', 'measurements_sorted'):
#             if return_type == 'self':
#                 m = self
#                 m.clear()
#             if return_type == 'self_type':
#                 m = type(self)()
#             if return_type == 'measurements_unsorted':
#                 m = MeasurementsUnsorted()
#             if return_type == 'measurements_sorted':
#                 m = MeasurementsSorted()
#             
#             for i in range(len(points)):
#                 m.add_result(points[i], values[i])
# #             m.add_results(points, values)
#             
#             obj = m
#         
#         ## return array
#         if return_type == 'array':
#             def get_value_len(value):
#                 try:
#                     return len(value)
#                 except TypeError:
#                     return 1
#             value_ref_len = get_value_len(values[0])
#             for value in values:
#                 if get_value_len(value) != value_ref_len:
#                     raise ValueError('Len of each element in values has to be the same, but a len is {}!'.format(get_value_len(value)))
#             
#             n = len(points)
#             m = len(points[0]) + value_ref_len
#             
#             array = np.empty((n, m))
#             for i in range(n):
#                 array[i, :-value_ref_len] = points[i]
#                 array[i, -value_ref_len:] = values[i]
#             
#             obj = array
#         
#         
#         logger.debug('Returning {} values as type {}.'.format(len(points), type(obj)))
#         return obj
#         
#             
#         
#     
# #     def all_values(self):
# #         measurements_dict = self.measurements_dict
# #         all_measurements = []
# #         all_results = []
# #         
# #         ## iterate
# #         for (t, t_dict) in measurements_dict.items():
# #             for (x, x_dict) in t_dict.items():
# #                 for (y, y_dict) in x_dict.items():
# #                     for (z, results_list) in y_dict.items():
# #                         if len(results_list) >= minimum_measurements:
# #                             measurement = (t, x, y, z)
# #                             all_measurements.append(measurement)
# #                             results = np.array(results_list)
# #                             all_results.append(results)
#         
# #         logger.debug('Returning {} values of measurements with minimal {} results.'.format(len(all_results), minimum_measurements))
#     
#     
#     
#     
#     
#     
#     ## transform indices
#     
#     def transform_indices(self, transform_function):
#         measurements_dict = self.measurements_dict
#         self.measurements_dict = type(measurements_dict)()
#         
#         for (point, result_list) in self._iterate_generator_measurements_dict(measurements_dict):
#             point_transformed = transform_function(point)
#             self.extend_result_list(point_transformed, result_list)
#         
# #         for (t, t_dict) in  measurements_dict.items():
# #             for (x, x_dict) in t_dict.items():
# #                 for (y, y_dict) in x_dict.items():
# #                     for (z, results) in y_dict.items():
# #                         index = (t, x, y, z)
# #                         index_transformed = transform_function(index)
# #                         self.add_result(index_transformed, results)
#                     
#     
#     
#     
#     
#     def categorize_indices(self, separation_values, discard_year=False):
#         def categorize_index(index, separation_values, discard_year=False):
#             index = list(index)
#             
#             ## discard year
#             if discard_year:
#                 index[0] = index[0] % 1
#             
#             ## remove right bound of last y box
#             if index[2] == 90:
#                 index[2] = 90 - 10**(-6)
#             
#             ## iterate over dimensions
#             for i in range(len(separation_values)):
#                 
#                 ## get separation value
#                 try:
#                     separation_value = separation_values[i]
#                 except IndexError:
#                     separation_value = None
#                 
#                 ## categorize dimension i
#                 if separation_value is not None:
#                     # check if sequence
#                     try:
#                         is_sequence = len(separation_value) >= 2
#                     except TypeError:
#                         is_sequence = False
#                     
#                     # separate by value
#                     if not is_sequence:
#                         index[i] = (math.floor(index[i] / separation_value) + 0.5) * separation_value
#                     # separate by sequence
#                     else:
#                         index_right = bisect.bisect_right(separation_value, index[i])
#                         value_left = separation_value[index_right - 1]
#                         try:
#                             value_right = separation_value[index_right]
#                         except IndexError:
#                             raise ValueError('Index value %d exceeds range of separation values (right sight of separation values is %d).' % (index[i], value_left))
#                         index[i] = (value_left + value_right) / 2.0
#             
#             index = tuple(index)
#             return index
#         
#         
#         if discard_year:
#             logger.debug('Indices categorized by separation values %s and discard year.' % str(separation_values))
#         else:
#             logger.debug('Indices categorized by separation values %s.' % str(separation_values))
#         
# #         transform_function = lambda index: categorize_index(index, separation_values, wrap_around_ranges=wrap_around_ranges, discard_year=discard_year)
#         transform_function = lambda index: categorize_index(index, separation_values, discard_year=discard_year)
#         
#         self.transform_indices(transform_function)
#     
#     
#     def categorize_indices_to_lsm(self, lsm, discard_year=False):
#         self.categorize_indices((1/lsm.t_dim, 360/lsm.x_dim, 180/lsm.y_dim, lsm.z), discard_year=discard_year)
#     
#     
#     
#     def transform_indices_to_boxes(self, x_dim, y_dim, z_values_left):
#         def transform_index_to_boxes(index, x_dim, y_dim, z_values_left):
#             def transform_space_index(index, range, new_len):
#                 index = np.floor(((index - range[0]) / (range[1] - range[0])) * new_len)
#                 if index == new_len:
#                     index -= 1
#                 return index
#             
#             index = list(index)
#             
#             if index[1] < 0:
#                 index[1] += 360
#             index[1] = transform_space_index(index[1], (0, 360), x_dim)
#             index[2] = transform_space_index(index[2], (-90, 90), y_dim)
#             index[3] = bisect.bisect_right(z_values_left, index[3]) - 1
#             
#             index = tuple(index)
#             
#             return index
#         
#         
#         logger.debug('Transform indices to boxes with x_dim {}, y_dim {} and z_values_left {}.'.format(x_dim, y_dim, z_values_left))
#         
#         transform_function = lambda index: transform_index_to_boxes(index, x_dim=x_dim, y_dim=y_dim, z_values_left=z_values_left)
#         
#         self.transform_indices(transform_function)
#     
#     
#     def transform_indices_to_lsm(self, lsm):
#         def transform_t(index):
#             index = list(index)
#             index[0] = index[0] % 1
#             index[0] = math.floor(index[0] * lsm.t_dim)
#             return index
#         
#         self.transform_indices(transform_t)
#         self.transform_indices_to_boxes(lsm.x_dim, lsm.y_dim, lsm.z_left)
#     
#     
#     def coordinates_to_map_indices(self, lsm):
#         self.transform_indices(lambda point: lsm.coordinate_to_map_index(point))
#     
#     def map_indices_to_coordinates(self, lsm):
#         self.transform_indices(lambda index: lsm.map_index_to_coordinate(index))
#     
#     
#     def discard_year(self):
#         logger.debug('Discarding year.')
#         
#         def transform_function(index):
#             index_list = list(index)
#             index_list[0] = index[0] % 1
#             index = tuple(index_list)
#             return index
#             
#         self.transform_indices(transform_function)
#     
#     
#     def dicard_index(self, index):
#         logger.debug('Discarding index {}.'.format(index))
#         
#         def transform_function(current_index):
#             current_index = list(current_index)
#             current_index[index] = 0
#             current_index = tuple(current_index)
#             return current_index
#             
#         self.transform_indices(transform_function)
#     
#     
#     def dicard_indices(self, indices):
#         for index in indices:
#             self.dicard_index(index)
#     
#     
#     def discard_time(self):
#         logger.debug('Discarding time.')
#         self.dicard_index(0)
#     
#     def discard_space(self):
#         logger.debug('Discarding space.')
#         self.dicard_indices((1,2,3))
#     
#     
#     
#     
#     ## transform results
#     
#     def transform_result(self, transform_function):
#         measurements_dict = self.measurements_dict
#         
#         for (t, t_dict) in  measurements_dict.items():
#             for (x, x_dict) in t_dict.items():
#                 for (y, y_dict) in x_dict.items():
#                     for (z, result_list) in y_dict.items():
#                         index = (t, x, y, z)
#                         transformed_result_list = []
#                         for result in result_list:
#                             transformed_result = transform_function(index, result)
#                             transformed_result_list.append(transformed_result)
#                         y_dict[z] = transformed_result_list
# #                         results_transformed = transform_function(index, results)
# #                         
# #                         # make to list if not a list
# #                         try:
# #                             results_transformed = list(results_transformed)
# #                         except TypeError:
# #                             results_transformed = [results_transformed]
# #                         y_dict[z] = results_transformed
#     
#     
#     
#     def normalize(self, same_bounds, minimum_measurements=5):
#         logger.debug('Normalizing results with same bounds %s and min measurements %d.' % (same_bounds, minimum_measurements))
#         
#         ## save measurements dict
#         measurements_dict = self.measurements_dict
#         
#         ## get means and deviations
#         self.categorize_indices(same_bounds, discard_year=True)
#         means = self.means(minimum_measurements=minimum_measurements, return_type='measurments_unsorted')
#         deviations = self.deviations(minimum_measurements=minimum_measurements, return_type='measurments_unsorted')
#         
#         ## prepare new measurements dict
#         self.measurements_dict = type(measurements_dict)()
#         
#         ## iterate
#         for (t, t_dict) in measurements_dict.items():
#             for (x, x_dict) in t_dict.items():
#                 for (y, y_dict) in x_dict.items():
#                     for (z, results_list) in y_dict.items():
#                         index = (t, x, y, z)
#                         categorized_index = self.categorize_index(index, same_bounds, discard_year=True)
#                         
#                         try:
#                             mean = means[categorized_index][0]
#                             deviation = deviations[categorized_index][0]
#                             match = True
#                         except KeyError:
#                             match = False
#                         
#                         if match and deviation > 0:
#                             for result in results_list:
#                                 result_normalized = (result - mean) / deviation
#                                 self.add_result(index, result_normalized)
#     
#     
#     def set_min_result(self, min_result):
#         logger.debug('Applying min value {} to results.'.format(min_result))
#         
# #         def transform_function(index, results):
# #             for i in range(len(results)):
# #                 if results[i] < min_results:
# #                     results[i] = min_results
# #             return results
#         transform_function = lambda index, result: max([result, min_result])
#         self.transform_result(transform_function)
#         
#     
#     
#     def log_results(self):
#         logger.debug('Applying logarithm to results.')
# #         transform_function = lambda index, result: (np.log(np.asarray(results))).tolist()
#         transform_function = lambda index, result: np.log(result)
#         self.transform_result(transform_function)
#     
#     
#     ## filter
#     
#     def filter(self, filter_function, apply_to_copy=False):
#         if apply_to_copy:
#             measurements_base = self
#             measurements_results = type(self)()
#         else:
#             measurements_base = type(self)()
#             measurements_results = self
#             
#             ## swap measurement dicts
#             measurements_dict_tmp = measurements_base.measurements_dict
#             measurements_base.measurements_dict = measurements_results.measurements_dict
#             measurements_results.measurements_dict = measurements_dict_tmp
#         
#         ## iterate over measurements
#         for (point, results) in measurements_base.iterator_keys_and_value_lists():
#             if filter_function(point, results):
#                 measurements_results.set_result_list(point, results)
#         
#         return measurements_results
#     
#     
#     def filter_min_measurements(self, min_measurements=1, apply_to_copy=False):
#         def filter_function(index, results):
#             return len(results) >= min_measurements
#         
#         return self.filter(filter_function, apply_to_copy=apply_to_copy)
#     
#     
#     def filter_same_point_with_bounds(self, point, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True):
#         ## equal_bounds is float -> copy values for each dim
#         try:
#             float(equal_bounds)
#             equal_bounds = [equal_bounds]*4
#         except TypeError:
#             pass
#         
#         ## search equal_bound for z if z has sequence
#         assert len(equal_bounds) == 4
#         equal_bounds = list(equal_bounds)
#         try:
#             float(equal_bounds[3])
#         except TypeError:
#             ## get z bound for point z value
#             z_bound_found = False
#             i = 0
#             while not z_bound_found:
#                 if point[3] <= equal_bounds[3][i+1, 0]:
#                     z_bound_found = True
#                 else:
#                     i += 1
#                     z_bound_found = i == len(equal_bounds[3]) - 1
#             equal_bounds[3] = equal_bounds[3][i, 1]
#         equal_bounds = np.array(equal_bounds)
#         
#         ## prepare point
#         point_base = np.array(point, copy=True)
#         if discard_year:
#             point_base[0] = point_base[0] % 1
#         
#         
#         ## filter all measurements with point in equal bounds
#         filtered_points = []
#         filtered_results = []
#         
#         for (t, t_dict) in  self.measurements_dict.items():
#             if (not discard_year and np.abs(point_base[0] - t) <= equal_bounds[0]) or (discard_year and np.abs(point_base[0] - t%1) <= equal_bounds[0]):
#                 for (x, x_dict) in t_dict.items():
#                     if np.abs(point_base[1] - x) <= equal_bounds[1]:
#                         for (y, y_dict) in x_dict.items():
#                             if np.abs(point_base[2] - y) <= equal_bounds[2]:
#                                 for (z, result_list) in y_dict.items():
#                                     if np.abs(point_base[3] - z) <= equal_bounds[3]:
#                                         point = (t, x, y, z)
#                                         for result in result_list:
#                                             filtered_points.append(point)
#                                             filtered_results.append(result)
#         
#         ## filter only one per year
#         measurements_filtered = type(self)()
#         if only_one_per_year:
#             filtered_points = np.array(filtered_points)
#             filtered_results = np.array(filtered_results)
#             
#             years = np.unique(np.floor_divide(filtered_points[:,0], 1))
#             point_scale = np.copy(point_base)
#             for year in years:
#                 point_scale[0] = point_base[0]%1 + year
#                 
#                 min_index = np.linalg.norm(((filtered_points - point_scale) / equal_bounds), ord=2, axis=1).argmin()
#                 measurements_filtered.add_result(filtered_points[min_index], filtered_results[min_index])
#             number_of_filtered_measurements = len(years)
#         else:
# #             measurements_filtered.add_results(filtered_points, filtered_results)
#             for i in range(len(filtered_points)):
#                 measurements_filtered.add_result(filtered_points[i], filtered_results[i])
#             number_of_filtered_measurements = len(filtered_results)
#         
#         logger.debug('{} results for point {} with equal bounds {} filtered.'.format(number_of_filtered_measurements, point, equal_bounds))
#         
#         return measurements_filtered
#     
#     
#     def filter_same_point_except_year(self, point, return_type='measurements_unsorted'):
#         ## prepare point
#         point = list(point)
#         
#         ## filter all measurements with same point
#         filtered_points = []
#         filtered_results = []
#         
#         for (t, t_dict) in  self.measurements_dict.items():
#             if point[0]%1 == t%1:
#                 point = point.copy()
#                 point[0] = t
#                 try:
#                     x_dict = t_dict[point[1]]
#                     y_dict = x_dict[point[2]]
#                     result_list = y_dict[point[3]]
#                 except KeyError:
#                     result_list = None
#                 
#                 if result_list is not None:
#                     for result in result_list:
#                         filtered_points.append(point)
#                         filtered_results.append(result)
#         
#         logger.debug('{} results for point {} filtered.'.format(len(filtered_results), point))
#         return self._return_points_and_values_as_type(filtered_points, filtered_results, return_type=return_type)
#     
#     
#     def filter_same_points_with_same_function(self, filter_same_point_function, min_measurements=10, return_type='measurements'):
#         assert callable(filter_same_point_function)
#         
#         return_points = []
#         return_values = []
#         
#         
#         for (point, results) in self.iterator_keys_and_value_lists():
#             point = list(point)
#             point[0] = point[0] % 1
#             
#             if not point in return_points:
#                 measurements_same_point = filter_same_point_function(point)
#                 
# #                 transform_function = lambda point, results: [(point[0], result) for result in results]
#                 transform_function = lambda point, result: (point[0], result)
#                 measurements_same_point.transform_result(transform_function)
#                 measurements_same_point_results = measurements_same_point.all_results()
#                 
#                 if len(measurements_same_point_results) >= min_measurements:
#                     logger.debug('{} values with point {} passed filter.'.format(len(measurements_same_point_results), point))
#                     return_points.append(point)
#                     return_values.append(measurements_same_point_results)
#         
#         return self._return_points_and_values_as_type(return_points, return_values, return_type=return_type)
#     
#     
#     def filter_same_points_except_year(self, min_measurements=10, return_type='measurements'):
#         logger.debug('Filtering results with same indicies with min measurements {}.'.format(min_measurements))
#         
#         filter_same_point_function = lambda point: self.filter_same_point_except_year(point, return_type='measurements_unsorted')
#         return self.filter_same_points_with_same_function(filter_same_point_function, min_measurements=min_measurements, return_type=return_type)
#     
#     
#     def filter_same_points_with_bounds(self, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True, min_measurements=10, return_type='measurements'):
#         logger.debug('Filtering results with same indicies with equal bound {}, discard year {} and min measurements {}.'.format(equal_bounds, discard_year, min_measurements))
#         
#         filter_same_point_function = lambda point: self.filter_same_point_with_bounds(point, equal_bounds=equal_bounds, discard_year=discard_year, only_one_per_year=only_one_per_year)
#         return self.filter_same_points_with_same_function(filter_same_point_function, min_measurements=min_measurements, return_type=return_type)
#         
#     
#     
#     
# #     def filter_same_indices(self, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True, min_measurements=10, return_type='measurements'):
# #         logger.debug('Filtering results with same indicies with equal bound {}, discard year {} and min measurements {}.'.format(equal_bounds, discard_year, min_measurements))
# #         
# #         return_points = []
# #         return_values = []
# #         
# #         for (t, t_dict) in  self.measurements_dict.items():
# #             if discard_year:
# #                 t = t % 1
# #             for (x, x_dict) in t_dict.items():
# #                 for (y, y_dict) in x_dict.items():
# #                     for (z, results) in y_dict.items():
# #                         index = (t, x, y, z)
# #                         
# #                         if not return_measurements.has_result_list(index):
# #                             measurements_same_index = self.filter_same_index(index, equal_bounds=equal_bounds, discard_year=discard_year, only_one_per_year=only_one_per_year, apply_to_copy=True)
# #                             
# #                             transform_function = lambda index, results: [(index[0], result) for result in results]
# #                             measurements_same_index.transform_result(transform_function)
# #                             measurements_same_index_results = measurements_same_index.all_results()
# #                             
# #                             if len(measurements_same_index_results) >= min_measurements:
# #                                 logger.debug('{} values with index {} passed filter.'.format(len(measurements_same_index_results), index))
# #                                 return_points.append(index)
# #                                 return_values.append(measurements_same_index_results)
# #         
# #         return self._return_points_and_values_as_type(return_points, return_values, return_type=return_type)
#     
#     
#     ## compute values
#     
#     def numbers(self, minimum_measurements=1, return_type='array'):
#         logger.debug('Calculate numbers of measurements with %d minimal measurements.', minimum_measurements)
#         
#         return self.iterate_results(len, minimum_measurements, return_type=return_type)
#     
#     
#     def means(self, minimum_measurements=1, return_type='array'):
#         logger.debug('Calculate means of measurements with %d minimal measurements.', minimum_measurements)
#         
#         return self.iterate_results(np.average, minimum_measurements, return_type=return_type)
#     
#     
#     def variances(self, minimum_measurements=3, min_variance=0, return_type='array'):
#         logger.debug('Calculate variances of measurements with %d minimal measurements.', minimum_measurements)
#         
#         def calculate_variance(results):
#             mean = np.average(results)
#             number_of_results = results.size
#             variance = np.sum((results - mean)**2) / (number_of_results - 1)
#             variance = max([variance, min_variance])
#             return variance
#         
#         return self.iterate_results(calculate_variance, minimum_measurements, return_type=return_type)
#     
#     
#     def deviations(self, minimum_measurements=3, min_deviation=0, return_type='array'):
#         logger.debug('Calculate standard deviations of measurements with %d minimal measurements.', minimum_measurements)
#         
#         def calculate_deviation(results):
#             mean = np.average(results)
#             number_of_results = results.size
#             deviation = (np.sum((results - mean)**2) / (number_of_results - 1))**(1/2)
#             deviation = max([deviation, min_deviation])
#             return deviation
#         
#         return self.iterate_results(calculate_deviation, minimum_measurements, return_type=return_type)
#     
#     
#     
#     def correlation(self, min_measurements=10):
#         measurements_correlation = type(self)()
#         
#         ## iterate over each pair of measurement indices
#         index_of_measurement1 = 0
#         for (point1, transformed_results1) in self.iterator_keys_and_value_lists():
#             index_of_measurement2 = 0
#             for (point2, transformed_results2) in self.iterator_keys_and_value_lists():
#                 
#                 ## skip previous measurements
#                 if index_of_measurement2 > index_of_measurement1:
#                     
#                     ## get matching measurements
#                     res = []
#                     for (t1, r1) in transformed_results1:
#                         for (t2, r2) in transformed_results2:
#                             t_diff = np.abs((t1 - (point1[0]%1)) - (t2 - (point2[0]%1)))
#                             if t_diff < 0.9: # could also be 1.0, but 0.9 for rounding errors
#                                 res.append([r1, r2])
#                     res = np.array(res)
#                     n = len(res)
#                     
#                     ## calculate correlation
#                     if n >= min_measurements:
#                         x1 = res[:,0]
#                         x2 = res[:,1]
#                         
#                         m1 = x1.mean()
#                         m2 = x2.mean()
#                         s1 = np.sqrt(np.sum((x1 - m1)**2))
#                         s2 = np.sqrt(np.sum((x2 - m2)**2))
#                         
#                         correlation = np.sum((x1 - m1) / s1 * (x2 - m2) / s2)
#                         correlation_index = np.abs(np.asarray(point1) - np.asarray(point2))
#                         
#                         measurements_correlation.add_result(correlation_index, (n, correlation))
#                         
#                         logger.debug('Correlation {} calculated with {} measurements for index {}.'.format(correlation, n, correlation_index))
#                 
#                 index_of_measurement2 += 1
#             index_of_measurement1 += 1
#         
#         return measurements_correlation
#     
#     
#     
#     ## tests for normality
#     
#     def dagostino_pearson_test(self, minimum_measurements=50, alpha=0.05, return_type='array'):
#         logger.debug('Calculate D´Agostino-Person-test for normality of measurements with minimal {} results with alpha {}.'.format(minimum_measurements, alpha))
#         
#         test_results = self.iterate_results(lambda x: scipy.stats.normaltest(x)[1], minimum_measurements, return_type=return_type)
#         
#         if alpha is not None:
#             test_results[:,4] = (test_results[:,4] >= alpha).astype(np.float)
#         return test_results
#     
#     def shapiro_wilk_test(self, minimum_measurements=50, alpha=0.05, return_type='array'):
#         logger.debug('Calculate Shapiro-Wilk-test for normality of measurements with minimal {} results with alpha {}.'.format(minimum_measurements, alpha))
#         
#         test_results = self.iterate_results(lambda x: scipy.stats.shapiro(x)[1], minimum_measurements, return_type=return_type)
#         
#         if alpha is not None:
#             test_results[:,4] = (test_results[:,4] >= alpha).astype(np.float)
#         return test_results
#     
#     def anderson_darling_test(self, minimum_measurements=50, alpha=0.05, return_type='array'):
#         logger.debug('Calculate Anderson-Darling-test for normality of measurements with minimal {} results with alpha {}.'.format(minimum_measurements, alpha))
#         
#         def test(x, alpha):
#             ## get test values
#             t = scipy.stats.anderson(x)
#             test_value = t[0]
#             test_bounds = t[1]
#             test_alphas = t[2] / 100
#             
#             ## get bound for alpha
#             index = np.where(test_alphas == alpha)[0]
#             if len(index) == 0:
#                 raise ValueError('The alpha value {} is not supported for this test. Only to values {} are supported.'.format(alpha, test_alphas))
#             index = index[0]
#             bound = test_bounds[index]
#             
#             ## check if test passed
#             return test_value <= bound
#         
#         test_results = self.iterate_results(lambda x:test(x, alpha), minimum_measurements, return_type=return_type)
#         return test_results
#         
#     
#     
#     ## total correlogram and correlation (autocorrelation)
#     
#     def _get_first_dim_shifted(self, measurements_dict_list, shift, same_bound, wrap_around_range=None):
#         logger.debug('Getting first dim shifted with shift %f and same bound %f.' % (shift, same_bound))
#         
#         if wrap_around_range is not None:
#             wrap_around_len = wrap_around_range[1] - wrap_around_range[0]
#         
#         shift_list = []
#         
#         ## iterate over all dicts
#         for (measurements_dict, measurements_dict_shifted) in measurements_dict_list:
#             
#             ## iterate over all unshifted
#             for (i, i_dict) in measurements_dict.items():
#                 i_shifted_desired = i + shift
#                 if wrap_around_range is not None:
#                     i_shifted_desired = measurements.util.calculate.wrap_around_index(i_shifted_desired, wrap_around_range)
#                 
#                 ## iterate over all shifted
#                 for (i_shifted, i_dict_shifted) in measurements_dict_shifted.items():
#                     i_diff = abs(i_shifted - i_shifted_desired)
#                     if wrap_around_range is not None:
#                         i_diff = min(i_diff, wrap_around_len - i_diff)
#                     
#                     ## insert unshifted in shifted in shift_list
#                     if i_diff <= same_bound:
#                         shift_list.append((i_dict, i_dict_shifted))
#         
#         return shift_list
#     
#     
#     
#     def _get_results_together_with_shifted(self, factor, direction, same_bounds, x_range, t_range=None):
#         logger.debug('Gathering results with direction %s shifted by factor %f with same bound %s.' % (direction, factor, same_bounds))
#         
#         measurements_dict_list = [(self.measurements_dict, self.measurements_dict)]
#         dim = len(direction)
#         wrap_around_range = (t_range, x_range, None, None)
#         
#         ## iterate over dim and search matching shifts
#         for i in range(dim):
#             measurements_dict_list = self._get_first_dim_shifted(measurements_dict_list, factor * direction[i], same_bounds[i], wrap_around_range=wrap_around_range[i])
#         
#         logger.debug('Results gathered.')
#         
#         return measurements_dict_list
#     
#     
#     
#     def _get_array_from_shift_list(self, shift_list):
#         ## calculate length
# #         logger.debug('Getting array from shift list.')
#         n = 0
#         for (result_list, result_shifted_list) in shift_list:
#             n += len(result_list) * len(result_shifted_list)
#         
#         ## set values
# #         logger.debug('Creating array from shift list with length %d.' % n)
#         array = np.empty((n, 2))
#         i = 0
#         for (result_list, result_shifted_list) in shift_list:
#             for result in result_list:
#                 for result_shifted in result_shifted_list:
#                     array[i, 0] = result
#                     array[i, 1] = result_shifted
#                     i += 1
#         
#         assert i == n
# #         if n != i:
# #             warning('False array size: n=%d and i=%d' % (n, i))
# #         logger.debug('%d elements inserted. Returning array from shift list.' % i)
#         
#         return array
#     
#     
#     
#     def _calculate_total_correlation_from_shift_list(self, shift_list, is_normalized=False):
#         if not is_normalized:
#             #TODO mean and sd for each result list
#             shift_array = self._get_array_from_shift_list(shift_list)
#     #         shift_array = np.array(shift_list)
#             number = shift_array.shape[0]
#             
# #             logger.debug('Calulating correlation from %d pairs.' % number)
#             
#             x = shift_array[:,0]
#             y = shift_array[:,1]
#             
#             mean_x = np.average(x)
#             mean_y = np.average(y)
#             sd_x = np.sum((x - mean_x)**2)**(1/2)
#             sd_y = np.sum((y - mean_y)**2)**(1/2)
#             prod_array = ((x - mean_x) * (y - mean_y)) / (sd_x * sd_y)
#             
#             correlation = np.sum(prod_array)
#         else:
#             number = 0
#             correlation = 0
#             
#             for (result_list, result_shifted_list) in shift_list:
#                 for result in result_list:
#                     for result_shifted in result_shifted_list:
#                         correlation += result * result_shifted
#                         number += 1
#             
#             if number >= 1:
#                 correlation /= number
#             else:
#                 correlation = np.nan
#             
#         
#         logger.debug('Correlation %f calculated from %d measurements.' % (correlation, number))
#         
#         return (correlation, number)
#     
#     
#     
#     
#     
# #     def _calculate_correlations_from_shift_list(self, shift_list):
# # #         if not is_normalized:
# # #             #TODO mean and sd for each result list
# # #             shift_array = self._get_array_from_shift_list(shift_list)
# # #     #         shift_array = np.array(shift_list)
# # #             number = shift_array.shape[0]
# # #             
# # # #             logger.debug('Calulating correlation from %d pairs.' % number)
# # #             
# # #             x = shift_array[:,0]
# # #             y = shift_array[:,1]
# # #             
# # #             mean_x = np.average(x)
# # #             mean_y = np.average(y)
# # #             sd_x = np.sum((x - mean_x)**2)**(1/2)
# # #             sd_y = np.sum((y - mean_y)**2)**(1/2)
# # #             prod_array = ((x - mean_x) * (y - mean_y)) / (sd_x * sd_y)
# # #             
# # #             correlation = np.sum(prod_array)
# # #         else:
# #         number = 0
# #         correlation = 0
# #         
# #         for (result_list, result_shifted_list) in shift_list:
# #             xs = np.array(result_list)
# #             ys = np.array(result_shifted_list)
# #             
# #             mean_x = np.average(xs)
# #             mean_y = np.average(ys)
# #             sd_x = np.sum((xs - mean_x)**2)**(1/2)
# #             sd_y = np.sum((ys - mean_y)**2)**(1/2)
# #             
# #             for x in xs:
# #                 for y in ys:
# #                     correlation += (x - mean_x) * (y - mean_y)
# #                     number += 1
# #             prod_array = ((x - mean_x) * (y - mean_y))
# #             correlation = np.sum(prod_array) / (sd_x * sd_y)
# #         
# #         if number >= 1:
# #             correlation /= number
# #         else:
# #             correlation = np.nan
# #             
# #         
# #         logger.debug('Correlation %f calculated from %d measurements.' % (correlation, number))
# #         
# #         return (correlation, number)
#     
#     
#     
#     def _iterate_over_shift_in_direction(self, calculate_function, direction, same_bounds, dim_ranges, wrap_around_t=False, file=None):
#         logger.debug('Applying function to shifts by direction %s with same_bounds %s and dim_ranges %s.' % (direction, same_bounds, dim_ranges))
#         
#         ## init
#         function_results_list = []
#         direction_array = np.array(direction, dtype=np.float)
#         if wrap_around_t:
#             t_range = (dim_ranges[0][0], dim_ranges[0][1])
#         else:
#             t_range = None
#         x_range = (dim_ranges[1][0], dim_ranges[1][1])
#         
#         ## calculate max factor
#         if not np.all(direction == 0):
#             dim_ranges_array = np.array(dim_ranges, dtype=np.float)
#             dim_ranges_diff = dim_ranges_array[:,1] - dim_ranges_array[:,0]
#             if wrap_around_t:
#                 dim_ranges_diff[0] = dim_ranges_diff[0] / 2
#             dim_ranges_diff[1] = dim_ranges_diff[1] / 2
#             max_factor_mask = direction_array != 0
#             max_factor = math.floor(min(dim_ranges_diff[max_factor_mask] / direction_array[max_factor_mask]))
#         else:
#             max_factor = 0
#         
#         logger.debug('Max factor is %d.', max_factor)
#         
#         ## iterate over all factors
#         for factor in range(max_factor + 1):
#             shift_list = self._get_results_together_with_shifted(factor, direction, same_bounds, x_range, t_range)
#             
#             ## apply calculate_function to shift list
#             logger.debug('Applying calculate function to shifts.')
#             function_result = calculate_function(shift_list)
#             function_results_list.append(function_result)
#             
#             ## save intermediate result
#             if file is not None:
#                 function_results_array = np.array(function_results_list)
#                 np.save(file, function_results_array)
#         
#         ## create array and save results
#         function_results_array = np.array(function_results_list)
#         
#         if file is None:
#             np.save(file, function_results_array)
#         
#         logger.debug('Results array calculated.')
#         
#         return function_results_array
#     
#     
#     
#     def total_correlogram(self, direction, same_bounds, dim_ranges, wrap_around_t=False, minimum_measurements=1, is_normalized=False, file=None):
#         
#         logger.debug('Calculating correlogram.')
#         
#         calculate_correlation = lambda shift_list : self._calculate_total_correlation_from_shift_list(shift_list, is_normalized=is_normalized)
#         correlogram = self._iterate_over_shift_in_direction(calculate_correlation, direction, same_bounds, dim_ranges, wrap_around_t=wrap_around_t, file=file)
#         
#         logger.debug('Correlogram calculated.')
#         
#         return correlogram
#     
#     
#     
#     
#     
#     def _iterate_over_shift_all_factor_combinations(self, calculation_function, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=1, file=None):
#         
#         logger.debug('Iterate over all shifts with all factor combinations with the following configurations: direction=%s, factor_lists=%s, same_bounds=%s, wrap_around_ranges=%s, minimum_measurements=%d.' % (direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements))
#         
#         function_results_list = []
#         
#         dim = len(direction)
#         max_index = np.array([len(factor_list) for factor_list in factor_lists])
#         
#         measurements_dict_list = [[(self.measurements_dict, self.measurements_dict),],] + [None] * dim
#         current_dim = 0
#         current_indices = np.zeros(dim, dtype=np.int)
#         current_shift = np.zeros(dim)
#         
#         ## iterate over all factors
#         while current_dim >= 0:
#             
#             ## iterate over all dimensions
#             while 0 <= current_dim < dim:
#                 current_index = current_indices[current_dim]
#                 current_factor_list = factor_lists[current_dim]
#                 
#                 ## search matching shifts
#                 current_factor = current_factor_list[current_index]
#                 current_shift[current_dim] = direction[current_dim] * current_factor
#                 measurements_dict_list[current_dim + 1] = self._get_first_dim_shifted(measurements_dict_list[current_dim], current_shift[current_dim], same_bounds[current_dim], wrap_around_ranges[current_dim])
#                 
#                 ## increase current dim
#                 current_dim += 1
#             
#             
#             ## calculate value and append to list
# #             results_with_shifted_list = [(result, result_shifted) for (result_list, result_shifted_list) in measurements_dict_list[dim] for result in result_list for result_shifted in result_shifted_list]
# #             results_with_shifted_iterable = itertools.chain([itertools.product(result_list, result_shifted_list) for (result_list, result_shifted_list) in measurements_dict_list[dim]])
#             
# #             number_of_measurements = len(results_with_shifted_list)
# #             (value, number_of_measurements) = calculation_function(results_with_shifted_iterable)
#             (value, number_of_measurements) = calculation_function(measurements_dict_list[dim])
#             if number_of_measurements >= minimum_measurements:
# #                 (value, number_of_measurements) = calculation_function(results_with_shifted_list)
#                 function_results_list.append(current_shift.tolist() + [value, number_of_measurements])
#                 logger.debug('Value %f for shift %s calculated and inserted. %d matching measurements where found.' % (value, current_shift, number_of_measurements))
#                 
#                 ## save intermediate result
#                 if file is not None:
#                     function_results_array = np.array(function_results_list)
#                     np.save(file, function_results_array)
#             else:
#                 logger.debug('Value for shift %s not inserted. Only %d matching measurements where found.' % (current_shift, number_of_measurements))
#             
#             
#             ## increase index
#             current_dim -= 1
#             measurements_dict_list[current_dim + 1] = None
#             current_indices[current_dim] += 1
#             
#             while current_dim >= 0 and current_indices[current_dim] == len(factor_lists[current_dim]):
#                 measurements_dict_list[current_dim] = None
#                 current_indices[current_dim] = 0
#                 current_dim -= 1
#                 if current_dim >= 0:
#                     current_indices[current_dim] += 1
#         
#         ## create array and save results
#         function_results_array = np.array(function_results_list)
#         
#         if file is None:
#             np.save(file, function_results_array)
#         
#         logger.debug('Iterate over all shifts with all factor combinations finished.')
#         
#         return function_results_array
#     
#     
#     def total_correlation(self, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=1, is_normalized=False, file=None):
#         logger.debug('Calculating correlation with the following configurations: direction=%s, factor_lists=%s, same_bounds=%s, wrap_around_ranges=%s, minimum_measurements=%d.' % (direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements))
#         
#         calculate_correlation = lambda shift_list : self._calculate_total_correlation_from_shift_list(shift_list, is_normalized=is_normalized)
#         correlation = self._iterate_over_shift_all_factor_combinations(calculate_correlation, direction, factor_lists, same_bounds, wrap_around_ranges, minimum_measurements=minimum_measurements, file=file)
#         
#         logger.debug('Correlation calculated.')
#         
#         return correlation
#     
#     
#     
#     
# #     def filter_same_indices(self, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True, min_measurements=10, return_measurements=None):
# #         logger.debug('Filtering results with same indicies with equal bound {}, discard year {} and min measurements {}.'.format(equal_bounds, discard_year, min_measurements))
# #         
# #         if return_measurements is None:
# #             return_measurements = type(self)()
# #         
# #         for (t, t_dict) in  self.measurements_dict.items():
# #             if discard_year:
# #                 t = t % 1
# #             for (x, x_dict) in t_dict.items():
# #                 for (y, y_dict) in x_dict.items():
# #                     for (z, results) in y_dict.items():
# #                         index = (t, x, y, z)
# #                         
# #                         if not return_measurements.has_result_list(index):
# #                             measurements_same_index = self.filter_same_index(index, equal_bounds=equal_bounds, discard_year=discard_year, only_one_per_year=only_one_per_year, apply_to_copy=True)
# #                             
# #                             transform_function = lambda index, results: [(index[0], result) for result in results]
# #                             measurements_same_index.transform_result(transform_function)
# #                             measurements_same_index_results = measurements_same_index.all_results()
# #                             
# #                             if len(measurements_same_index_results) >= min_measurements:
# #                                 logger.debug('{} values with index {} passed filter.'.format(len(measurements_same_index_results), index))
# #                                 return_measurements.add_result(index, measurements_same_index_results)
# #         
# #         return return_measurements
# #     
# #     
# #     def filter_same_indices(self, equal_bounds=(0,0,0,0), discard_year=True, only_one_per_year=True, min_measurements=10, return_type='measurements'):
# #         logger.debug('Filtering results with same indicies with equal bound {}, discard year {} and min measurements {}.'.format(equal_bounds, discard_year, min_measurements))
# #         
# #         return_points = []
# #         return_values = []
# #         
# #         for (t, t_dict) in  self.measurements_dict.items():
# #             if discard_year:
# #                 t = t % 1
# #             for (x, x_dict) in t_dict.items():
# #                 for (y, y_dict) in x_dict.items():
# #                     for (z, results) in y_dict.items():
# #                         index = (t, x, y, z)
# #                         
# #                         if not return_measurements.has_result_list(index):
# #                             measurements_same_index = self.filter_same_index(index, equal_bounds=equal_bounds, discard_year=discard_year, only_one_per_year=only_one_per_year, apply_to_copy=True)
# #                             
# #                             transform_function = lambda index, results: [(index[0], result) for result in results]
# #                             measurements_same_index.transform_result(transform_function)
# #                             measurements_same_index_results = measurements_same_index.all_results()
# #                             
# #                             if len(measurements_same_index_results) >= min_measurements:
# #                                 logger.debug('{} values with index {} passed filter.'.format(len(measurements_same_index_results), index))
# #                                 return_points.append(index)
# #                                 return_values.append(measurements_same_index_results)
# #         
# #         return self._return_points_and_values_as_type(return_points, return_values, return_type=return_type)
#     
# #     
# #     
# #     def correlation(self, measurements_same_indices, min_measurements=10, equal_t_bound=0):
# #         measurements_correlation = type(self)()
# #         
# #         ## iterate over each pair of measurement indices
# #         number_of_measurement1 = 0
# #         for (point1, transformed_results1) in measurements_same_indices.iterator_keys_and_value_lists():
# #             number_of_measurement2 = 0
# #             for (point2, transformed_results2) in measurements_same_indices.iterator_keys_and_value_lists():
# #                 
# #                 ## skip previous measurements
# #                 if number_of_measurement2 > number_of_measurement1:
# #                     
# #                     ## get matching measurements
# #                     res = [(r1, r2) for (t1, r1) in transformed_results1 for (t2, r2) in transformed_results2 if np.abs(t1 - t2) <= equal_t_bound]
# #                     res = np.array(res)
# #                     n = len(res)
# #                     
# #                     ## calculate correlation
# #                     if n >= min_measurements:
# #                         x1 = res[:,0]
# #                         x2 = res[:,1]
# #                         
# #                         m1 = x1.mean()
# #                         m2 = x2.mean()
# #                         s1 = np.sqrt(np.sum((x1 - m1)**2))
# #                         s2 = np.sqrt(np.sum((x2 - m2)**2))
# #                         
# #                         correlation = np.sum((x1 - m1) / s1 * (x2 - m2) / s2)
# #                         correlation_index = np.abs(np.asarray(point1) - np.asarray(point2))
# #                         
# #                         measurements_correlation.add_result(correlation_index, (n, correlation))
# #                         
# #                         logger.debug('Correlation {} calculated with {} measurements for index {}.'.format(correlation, n, correlation_index))
# #                 
# #                 number_of_measurement2 += 1
# #             number_of_measurement1 += 1
# #         
# #         return measurements_correlation
#     
#     
#     
#     
#     
# #     def correlation(self, equal_bounds, min_measurements=10, measurements_same_indices=None):
# #         measurements_correlation = type(self)()
# #         
# #         if measurements_same_indices is None:
# #             measurements_same_indices = self.filter_same_indices(equal_bounds=equal_bounds, discard_year=True, only_one_per_year=True, min_measurements=min_measurements)
# #         
# #         ## iterate over each pair of measurement indices
# #         number_of_measurement1 = 0
# #         for (point1, transformed_results1) in measurements_same_indices.iterator_keys_and_value_lists():
# #             number_of_measurement2 = 0
# #             for (point2, transformed_results2) in measurements_same_indices.iterator_keys_and_value_lists():
# #                 
# #                 ## skip previous measurements
# #                 if number_of_measurement2 > number_of_measurement1:
# #                     
# #                     ## get matching measurements
# #                     res = [(r1, r2) for (t1, r1) in transformed_results1 for (t2, r2) in transformed_results2 if np.abs(t1 - t2) <= equal_bounds[0]]
# #                     res = np.array(res)
# #                     n = len(res)
# #                     
# #                     ## calculate correlation
# #                     if n >= min_measurements:
# #                         x1 = res[:,0]
# #                         x2 = res[:,1]
# #                         
# #                         m1 = x1.mean()
# #                         m2 = x2.mean()
# #                         s1 = np.sqrt(np.sum((x1 - m1)**2))
# #                         s2 = np.sqrt(np.sum((x2 - m2)**2))
# #                         
# #                         correlation = np.sum((x1 - m1) / s1 * (x2 - m2) / s2)
# #                         correlation_index = np.abs(np.asarray(point1) - np.asarray(point2))
# #                         
# #                         measurements_correlation.add_result(correlation_index, (n, correlation))
# #                         
# #                         logger.debug('Correlation {} calculated with {} measurements for index {}.'.format(correlation, n, correlation_index))
# #                 
# #                 number_of_measurement2 += 1
# #             number_of_measurement1 += 1
# #         
# #         return measurements_correlation
#             
#         
#         
# #         for (t1, t_dict1) in  measurements_same_indices.items():
# #             for (x1, x_dict1) in t_dict.items():
# #                 for (y1, y_dict1) in x_dict.items():
# #                     for (z1, transformed_results1) in y_dict.items():
# #                         index1 = np.array((t1, x1, y1, z1))
# #                         
# #                         for (t2, t_dict2) in  measurements_same_indices.items():
# #                             for (x2, x_dict2) in t_dict.items():
# #                                 for (y2, y_dict2) in x_dict.items():
# #                                     for (z2, transformed_results2) in y_dict.items():
# #                                         index2 = np.array((t2, x2, y2, z2))
# #                         
# #                         
# #                                         ## get matching measurements
# #                                         res = [(r1, r2) for (t1, r1) in transformed_results1 for (t2, r2) in transformed_results2 if np.abs(t1 - t2) <= equal_bounds[0]]
# #                                         res = np.array(res)
# #                                         n = len(res)
# #                                         
# #                                         ## calculate correlation
# #                                         if n >= min_measurements:
# #                                             x1 = res[:,0]
# #                                             x2 = res[:,1]
# #                                             
# #                                             m1 = x1.mean()
# #                                             m2 = x2.mean()
# #                                             s1 = np.sqrt(np.sum((x1 - m1)**2))
# #                                             s2 = np.sqrt(np.sum((x2 - m2)**2))
# #                                             
# #                                             correlation = np.sum((x1 - m1) / s1 * (x2 - m2) / s2)
# #                                             correlation_index = np.abs(index1 - index2)
# #                                             
# #                                             self.add_result(correlation_index, correlation)
#         
#     
#     
# 
# 
# 
# class MeasurementsSorted(MeasurementsUnsorted):
#     
#     def __init__(self):
#         from blist import sorteddict
#         self.measurements_dict = sorteddict()
#     
#     
#     def _get_first_dim_shifted(self, measurements_dict_list, shift, same_bound, wrap_around_range=None):
#         logger.debug('Getting first dim shifted with shift %f and same bound %f.' % (shift, same_bound))
#         
#         if wrap_around_range is not None:
#             wrap_around_len = wrap_around_range[1] - wrap_around_range[0]
#         
#         shift_list = []
#         
#         ## iterate over all dicts
#         for (measurements_dict, measurements_dict_shifted) in measurements_dict_list:
#             
#             keys_view_shifted = measurements_dict_shifted.keys()
#             keys_view_shifted_len = len(keys_view_shifted)
#             
#             ## iterate over all unshifted
#             for (key, value) in measurements_dict.items():
#                 
#                 ## calculate desired key bounds
#                 key_shifted_desired_lower_bound = measurements.util.calculate.wrap_around_index(key + shift - same_bound, wrap_around_range)
#                 key_shifted_desired_upper_bound = measurements.util.calculate.wrap_around_index(key + shift + same_bound, wrap_around_range)
#                 key_shifted_desired_lower_bound_index = keys_view_shifted.bisect_left(key_shifted_desired_lower_bound)
#                 key_shifted_desired_upper_bound_index = keys_view_shifted.bisect_right(key_shifted_desired_upper_bound)
#                 
# #                 logger.log(logging.DEBUG + 10, 'Getting shifts for key %f with lower bound %f and upper bound %f.' % (key, key_shifted_desired_lower_bound, key_shifted_desired_upper_bound))
#                 
#                 ## if desired keys are available
#                 if key_shifted_desired_lower_bound_index != key_shifted_desired_upper_bound_index:
#                     if key_shifted_desired_upper_bound_index >= keys_view_shifted_len or keys_view_shifted[key_shifted_desired_upper_bound_index] > key_shifted_desired_upper_bound:
#                         key_shifted_desired_upper_bound_index -= 1
#                     
#                     ## calculate desired key range
#                     if key_shifted_desired_lower_bound_index <= key_shifted_desired_upper_bound_index:
#                         key_shifted_desired_index_range = range(key_shifted_desired_lower_bound_index, key_shifted_desired_upper_bound_index + 1)
#                     else:
#                         key_shifted_desired_index_range = itertools.chain(range(key_shifted_desired_lower_bound_index, keys_view_shifted_len), range(0, key_shifted_desired_upper_bound_index + 1))
#                     
#                     ## insert values with shifted values
#                     for key_shifted_desired_index in key_shifted_desired_index_range:
#                         key_shifted = keys_view_shifted[key_shifted_desired_index]
#                         value_shifted = measurements_dict_shifted[key_shifted]
#                         shift_list.append((value, value_shifted))
#                         
# #                         logger.log(logging.DEBUG + 10, 'Value for key %f inserted.' % (key_shifted))
# #                 else:
# #                     logger.log(logging.DEBUG + 10, 'No values in key range found.')
#         
#         return shift_list
#     
#     
#     
# #     def save(self, file):
# #         super().save(file)
# #     
# #     
# #     def load(self, file):
# #         super().load(file)
# 
