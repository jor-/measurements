import numpy as np

import measurements.po4.wod.data.results
import measurements.util.data
import measurements.util.correlation

import util.cache
import util.logging
logger = util.logging.logger

from measurements.constants import CORRELATION_SAME_BOX, CORRELATION_QUANTITY_SAME_BOX



def measurements_same_points(min_values):
    from measurements.po4.wod.correlation.constants import VALUE_DIR, MEASUREMENTS_SAME_POINTS_FILENAME

    def calculate_function():
        m = measurements.po4.wod.data.results.Measurements.load()
        from measurements.po4.wod.constants import SAMPLE_LSM
        m.categorize_indices_to_lsm(SAMPLE_LSM, discard_year=False)
        m.means(return_type='self')
        return m.filter_same_points_except_year(min_values=min_values)

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsSamePoints.load)
    return cache.get_value(MEASUREMENTS_SAME_POINTS_FILENAME.format(min_values=min_values), calculate_function)



def sample_values(value_type, min_values, max_year_diff=float('inf')):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))

    from measurements.po4.wod.correlation.constants import VALUE_DIR, VALUES_MEASUREMENTS_FILENAME

    def calculate_function():
        ms = measurements_same_points(min_values=min_values)
        covariance = ms.correlation_or_covariance(value_type, min_values=min_values, stationary=False, max_year_diff=max_year_diff)
        return covariance

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache.get_value(VALUES_MEASUREMENTS_FILENAME.format(type=value_type, max_year_diff=max_year_diff, min_values=min_values), calculate_function)



def sample_values_transformed(value_type, min_values, max_year_diff=float('inf')):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))

    from measurements.po4.wod.correlation.constants import VALUE_DIR, VALUES_MEASUREMENTS_TRANSFORMED_FILENAME

    def calculate_function():
        from measurements.po4.wod.constants import SAMPLE_LSM

        value_dict = sample_values(value_type, min_values=min_values, max_year_diff=max_year_diff)
        # value_dict.transform_values(lambda key, value: value[1])
        value_dict.coordinates_to_map_indices(SAMPLE_LSM)
        value_dict.keys_to_int_keys(np.int32)
        # value_dict.means(min_values=1, return_type='self')

        return value_dict

    cache = util.cache.HDD_ObjectWithSaveCache(VALUE_DIR, measurements.util.data.MeasurementsCovariance.load)
    return cache.get_value(VALUES_MEASUREMENTS_TRANSFORMED_FILENAME.format(type=value_type, max_year_diff=max_year_diff, min_values=min_values), calculate_function)



def transform_key(key, discard_year=False):
    from measurements.po4.wod.constants import SAMPLE_LSM
    key = measurements.util.data.Measurements.categorize_index(key, SAMPLE_LSM.separation_values, discard_year=discard_year)
    key = SAMPLE_LSM.coordinate_to_map_index(*key, discard_year=False)
    key = tuple(np.array(np.round(key), dtype=np.int32))
    return key




class SampleCorrelation():

    def __init__(self, min_values, max_year_diff=float('inf'), same_box_correlation=CORRELATION_SAME_BOX, same_box_quantity=CORRELATION_QUANTITY_SAME_BOX, no_data_correlation=None, return_type=measurements.util.correlation.RETURN_CORRELATION):
        logger.debug('Preparing sample correlation with min_value {}, same_box_correlation {}, same_box_quantity {}, no_data_correlation {} and return type {}.'.format(min_values, same_box_correlation, same_box_quantity, no_data_correlation, return_type))

        # ## chose return index
        # if return_type == RETURN_QUANTITY:
        #     return_index = 0
        # elif return_type == RETURN_CORRELATION:
        #     return_index = 1
        # elif return_type == RETURN_QUANTITY_AND_CORRELATION:
        #     return_index = slice(2)
        # else:
        #     raise ValueError('Return type {} has to be in {}!'.format(return_type, (RETURN_QUANTITY, RETURN_CORRELATION, RETURN_QUANTITY_AND_CORRELATION)))
        # self.return_index = return_index
        self.return_index = measurements.util.correlation.chose_return_index(return_type)


        ## save values
        self.same_box_correlation = same_box_correlation
        self.same_box_quantity = same_box_quantity
        # self.max_correlation = max_correlation
        self.no_data_correlation = no_data_correlation

        from measurements.po4.wod.constants import SAMPLE_LSM
        self.lsm_separation_values = SAMPLE_LSM.separation_values
        self.lsm = SAMPLE_LSM

        ## get value dict
        self.value_dict = sample_values_transformed('correlation', min_values=min_values, max_year_diff=max_year_diff)


    def value(self, keys):
        assert len(keys) == 2

        ## if same point return 1
        if np.all(keys[0] == keys[1]):
            quantity = self.same_box_quantity
            correlation = 1

        ## remove min t and categorize
        else:
            keys = [list(keys[0]), list(keys[1])]
            year_min = min([int(keys[0][0]), int(keys[1][0])])

            for i in range(len(keys)):
                keys[i][0] = keys[i][0] - year_min
                # keys[i] = measurements.util.data.Measurements.categorize_index(keys[i], self.lsm.separation_values, discard_year=False)
                # keys[i] = self.lsm.coordinate_to_map_index(*keys[i], discard_year=False)
                # keys[i] = tuple(np.array(np.round(keys[i]), dtype=np.int32))
                keys[i] = transform_key(keys[i], discard_year=False)

            ## if same point return same box correlation
            if np.all(keys[0] == keys[1]):
                quantity = self.same_box_quantity
                correlation = self.same_box_correlation

            ## otherwise use sample correlation
            else:
                try:
                    sample_correlation = self.value_dict[keys]
                except KeyError:
                    sample_correlation = None
                    quantity = 0
                    correlation = self.no_data_correlation

                if sample_correlation is not None:
                    assert len(sample_correlation) == 1
                    quantity, correlation = sample_correlation[0]

        return (quantity, correlation)[self.return_index]



    #
    # def value(self, keys):
    #     assert len(keys) == 2
    #
    #     ## if same point return 1
    #     if np.all(keys[0] == keys[1]):
    #         quantity = same_box_quantity
    #         correlation = 1
    #         return 1
    #
    #     ## remove min t and categorize
    #     keys = [list(keys[0]), list(keys[1])]
    #     year_min = min([int(keys[0][0]), int(keys[1][0])])
    #
    #     for i in range(len(keys)):
    #         keys[i][0] = keys[i][0] - year_min
    #         keys[i] = measurements.util.data.Measurements.categorize_index(keys[i], self.lsm.separation_values, discard_year=False)
    #         keys[i] = self.lsm.coordinate_to_map_index(*keys[i], discard_year=False)
    #         keys[i] = tuple(np.array(np.round(keys[i]), dtype=np.int32))
    #
    #     ## if same point return same box correlation
    #     if np.all(keys[0] == keys[1]):
    #         return self.same_box_correlation
    #
    #     ## otherwise use sample correlation
    #     try:
    #         sample_correlation = self.value_dict[keys]
    #     except KeyError:
    #         sample_correlation = None
    #
    #     if sample_correlation is not None:
    #         assert len(sample_correlation) == 1
    #         # sample_correlation = sample_correlation[0]
    #         # if np.abs(sample_correlation) <= self.max_correlation:
    #         #     correlation = sample_correlation
    #         # else:
    #         #     correlation = self.max_correlation * np.sign(sample_correlation)
    #         correlation = sample_correlation[0]
    #     else:
    #         correlation = self.no_data_value
    #
    #     return correlation
    #

    def __getitem__(self, key):
        return self.value(key)


    @property
    def number_of_sample_values(self):
        return len(self.value_dict)

    def __len__(self):
        return self.number_of_sample_values


    @property
    def effective_max_year_diff(self):
        d = 0
        for key in self.value_dict.keys():
            d = max(d, np.ceil(np.abs(key[0,0] - key[1,0])))
        return d
