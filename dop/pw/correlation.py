import numpy as np

import measurements.util.correlation
import measurements.util.data


class CorrelationModel(measurements.util.correlation.Model):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        from measurements.dop.pw.constants import SAMPLE_LSM
        self.sample_lsm = SAMPLE_LSM


    def sample_value(self, keys):
        assert len(keys) == 2

        ## if same point return 1
        if np.all(keys[0] == keys[1]):
            quantity = self.same_box_quantity
            correlation = 1

        ## categorize
        else:
            keys = [list(keys[0]), list(keys[1])]

            for i in range(len(keys)):
                keys[i] = measurements.util.data.Measurements.categorize_index(keys[i], self.sample_lsm.separation_values, discard_year=False)
                keys[i] = self.sample_lsm.coordinate_to_map_index(*keys[i], discard_year=False)
                keys[i] = np.array(np.round(keys[i]), dtype=np.int32)

            ## if same point return same box correlation
            if np.all(keys[0] == keys[1]):
                quantity = self.same_box_quantity
                correlation = self.same_box_correlation

            ## otherwise return 0
            else:
                quantity = 0
                correlation = self.no_data_correlation

        return (quantity, correlation)

    @property
    def number_of_sample_values(self):
        return 0

    @property
    def effective_max_year_diff(self):
        return self.max_year_diff



def sample_values_transformed(value_type, min_values, max_year_diff):
    POSSIBLE_VALUE_TYPES = ('correlation', 'covariance')
    if value_type not in POSSIBLE_VALUE_TYPES:
        raise ValueError('Value type has to be in {} but it is {}.'.format(POSSIBLE_VALUE_TYPES, value_type))

    return measurements.util.data.MeasurementsCovariance()





# import numpy as np
#
# import measurements.util.data
#
# from measurements.dop.constants import CORRELATION_SAME_BOX, CORRELATION_MIN_ABS_VALUE, CORRELATION_MAX_ABS_VALUE
#
#
# class CorrelationModel():
#
#     def __init__(self, min_abs_correlation=CORRELATION_MIN_ABS_VALUE, max_abs_correlation=CORRELATION_MAX_ABS_VALUE):
#         from measurements.dop.pw.constants import SAMPLE_LSM
#         self.correlation_same_box = CORRELATION_SAME_BOX
#         self.SAMPLE_LSM = SAMPLE_LSM
#
#
#     def value(self, keys):
#         assert len(keys) == 2
#
#         ## if same point return 1
#         if np.all(keys[0] == keys[1]):
#             return 1
#
#         ## categorize
#         keys = [list(keys[0]), list(keys[1])]
#
#         for i in range(len(keys)):
#             keys[i] = measurements.util.data.Measurements.categorize_index(keys[i], self.SAMPLE_LSM.separation_values, discard_year=False)
#             keys[i] = self.SAMPLE_LSM.coordinate_to_map_index(*keys[i], discard_year=False)
#             keys[i] = np.array(np.round(keys[i]), dtype=np.int32)
#
#         ## if same point return same box correlation
#         if np.all(keys[0] == keys[1]):
#             return self.correlation_same_box
#
#         ## otherwise return 0
#         return 0
#
#
#     def __getitem__(self, key):
#         return self.value(key)
#
#
#
#     @property
#     def number_of_sample_values(self):
#         return 0
#
#     def __len__(self):
#         return self.number_of_sample_values
