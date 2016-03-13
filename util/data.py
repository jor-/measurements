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


    @staticmethod
    def categorize_index_to_lsm(index, lsm, discard_year=False):
        return Measurements.categorize_index(index, (1/lsm.t_dim, 360/lsm.x_dim, 180/lsm.y_dim, lsm.z), discard_year=discard_year)
        
    
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


    def coordinates_to_map_indices(self, lsm, float_indices=True):
        self.transform_keys(lambda point: lsm.coordinate_to_map_index(*point, float_indices=float_indices))

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

    def filter_year(self, year, return_type='self'):
        return self.filter_key_range(0, [year, year+1-10**(-10)], return_type=return_type)
    
    
    def filter_points_near_water(self, lsm, max_land_boxes=0, return_type='self'):
        self.filter_with_boolean_function(lambda point, values: lsm.is_point_near_water(point, max_land_boxes=max_land_boxes), return_type=return_type)
    


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

        ## prepare value measurement dict
        if stationary:
            value_measurements = MeasurementsCovarianceStationary()
        else:
            value_measurements = MeasurementsCovariance()
        
        ## calculate values
        if len(self) > 0:
    
            ## check max_year_diff
            if max_year_diff is None or max_year_diff == float('inf'):
                t = self.values()[:,0]
                max_year_diff = int(np.ceil(t.max() - t.min()))
                logger.debug('Using max_year_diff {}.'.format(max_year_diff))
    
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
                                        value = np.sum((x0 - m0) * (x1 - m1)) / (n-1)
    
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

    def coordinates_to_map_indices(self, lsm, float_indices=True):
        logger.debug('Transforming in {} coordinates to map indices of {}'.format(self, lsm))

        self._year_len = lsm.t_dim
        self.transform_keys(lambda keys: (lsm.coordinate_to_map_index(*keys[0], discard_year=False, float_indices=float_indices), lsm.coordinate_to_map_index(*keys[1], discard_year=False, float_indices=float_indices)))


    def map_indices_to_coordinates(self, lsm):
        logger.debug('Transforming in {} map indices of {} to coordinates'.format(self, lsm))

        self._year_len = 1
        self.transform_keys(lambda indices: (lsm.map_index_to_coordinate(*indices[0]), lsm.map_index_to_coordinate(*indices[1])))



    ## io
    def save(self, file):
        only_dict = self._year_len != 1
        super().save(file, only_dict=only_dict)



class MeasurementsCovarianceStationary(util.multi_dict.MultiDictDiffPointPairs, Measurements):

    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)


