import numpy as np

from measurements.constants import CORRELATION_QUANTITY_SAME_BOX, CORRELATION_SAME_BOX, CORRELATION_MIN_ABS_VALUE, CORRELATION_MAX_ABS_VALUE

import util.logging
logger = util.logging.logger


RETURN_QUANTITY = 'quantity'
RETURN_CORRELATION = 'correlation'
RETURN_QUANTITY_AND_CORRELATION = 'quantity_correlation'


def chose_return_index(return_type):
    if return_type == RETURN_QUANTITY:
        return_index = 0
    elif return_type == RETURN_CORRELATION:
        return_index = 1
    elif return_type == RETURN_QUANTITY_AND_CORRELATION:
        return_index = slice(2)
    else:
        raise ValueError('Return type {} has to be in {}!'.format(return_type, (RETURN_QUANTITY, RETURN_CORRELATION, RETURN_QUANTITY_AND_CORRELATION)))
    return return_index



class Model():

    def __init__(self, min_values, max_year_diff=float('inf'), no_data_correlation=None, same_box_quantity=CORRELATION_QUANTITY_SAME_BOX, same_box_correlation=CORRELATION_SAME_BOX, min_abs_correlation=CORRELATION_MIN_ABS_VALUE, max_abs_correlation=CORRELATION_MAX_ABS_VALUE, return_type=RETURN_CORRELATION):

        logger.debug('Initiating {} with min_values {}, max_year_diff {}, no_data_correlation {}, same_box_quantity {}, same_box_correlation {}, min_abs_correlation {}, max_abs_correlation {} and return_type {}.'.format(self, min_values, max_year_diff, no_data_correlation, same_box_quantity, same_box_correlation, min_abs_correlation, max_abs_correlation, return_type))

        ## save values
        self.min_values = min_values
        self.max_year_diff = max_year_diff
        self.no_data_correlation = no_data_correlation
        self.same_box_quantity = same_box_quantity
        self.same_box_correlation = same_box_correlation
        self.min_abs_correlation = min_abs_correlation
        self.max_abs_correlation = max_abs_correlation
        self.return_type = return_type
        self.return_index = chose_return_index(return_type)


    def sample_value(self, keys):
        raise NotImplementedError("Please implement this method")


    def value(self, keys):
        if len(keys) != 2:
            raise ValueError('Keys have to be a tuple of length 2, but its length is {}!'.format(len(keys)))

        ## if same point return 1
        if np.all(keys[0] == keys[1]):
            quantity = self.same_box_quantity
            correlation = 1

        ## get sample values
        else:
            quantity, correlation = self.sample_value(keys)

        ## apply bounds
        if quantity != self.same_box_quantity:
            if np.abs(correlation) < self.min_abs_correlation:
                correlation = 0
            else:
                correlation = np.sign(correlation) * min(np.abs(correlation), self.max_abs_correlation)

        return (quantity, correlation)[self.return_index]


    def __getitem__(self, key):
        return self.value(key)


    @property
    def number_of_sample_values(self):
        raise NotImplementedError("Please implement this method")

    def __len__(self):
        return self.number_of_sample_values


    @property
    def effective_max_year_diff(self):
        raise NotImplementedError("Please implement this method")
