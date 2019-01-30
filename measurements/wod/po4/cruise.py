import numpy as np

import util.logging

import measurements.wod.cruise
import measurements.wod.po4.constants


class CruisePO4(measurements.wod.cruise.Cruise):

    def _load_depths_and_values(self, file):
        # load values
        try:
            depths = self._load_value_from_file_with_variable_information_dict(
                file, measurements.wod.po4.constants.DEPTH_VARIABLE)
            data = self._load_value_from_file_with_variable_information_dict(
                file, measurements.wod.po4.constants.DATA_VARIABLE)
        except measurements.wod.cruise.MissingValueError as e:
            util.logging.warn(e.message)
            depths = []
            data = []
        else:
            util.logging.debug('{} measurements found'.format(len(data)))

            # remove invalid measurements
            if len(data) > 0:
                z_flags = self._load_value_from_file(file, measurements.wod.po4.constants.DEPTH_VARIABLE['flag_name'])
                z_valids = z_flags == measurements.wod.po4.constants.DEPTH_VARIABLE['flag_valid_value']

                data_flags = self._load_value_from_file(file, measurements.wod.po4.constants.DATA_VARIABLE['flag_name'])
                data_valids = data_flags == measurements.wod.po4.constants.DATA_VARIABLE['flag_valid_value']

                data_profile_flag = self._load_value_from_file(file, measurements.wod.po4.constants.DATA_VARIABLE['profile_flag_name'])
                data_profile_valid = data_profile_flag == measurements.wod.po4.constants.DATA_VARIABLE['profile_flag_valid_value']

                data_not_missing = data != measurements.wod.po4.constants.DATA_VARIABLE['missing_data_value']

                valid_mask = np.logical_and(z_valids, data_valids) * data_profile_valid
                valid_mask = np.logical_and(valid_mask, data_not_missing)

                depths = depths[valid_mask]
                data = data[valid_mask]

                # check values
                if np.any(data < 0):
                    util.logging.warn('Value in {} are lower then 0!'.format(file))
                    valid_mask = data >= 0
                    data = data[valid_mask]
                    depths = depths[valid_mask]

                if np.any(depths < 0):
                    util.logging.warn('Depth in {} is lower then 0!'.format(file))
                    depths[depths < 0] = 0

                util.logging.debug('{} valid measurements found'.format(len(data)))

        return depths, data
