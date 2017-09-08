import numpy as np

import util.logging

import measurements.wod.cruise
import measurements.wod.po4.constants


class CruisePO4(measurements.wod.cruise.Cruise):

    def _load_depths_and_values(self, file):
        # load values
        try:
            depths = self._load_value_from_file(
                file,
                measurements.wod.po4.constants.DEPTH_NAME,
                unit=measurements.wod.po4.constants.DEPTH_UNIT)
            data = self._load_value_from_file(
                file,
                measurements.wod.po4.constants.DATA_NAME,
                unit=measurements.wod.po4.constants.DATA_UNIT)
        except measurements.wod.cruise.MissingValueError as e:
            util.logging.warn(e.message)
            depths = []
            data = []
        else:
            util.logging.debug('{} measurements found'.format(len(data)))

            # remove invalid measurements
            if len(data) > 0:
                z_flags = self._load_value_from_file(file, measurements.wod.po4.constants.DEPTH_FLAGS_NAME)
                z_valids = z_flags == measurements.wod.po4.constants.DEPTH_VALID_FLAG

                data_flags = self._load_value_from_file(file, measurements.wod.po4.constants.DATA_FLAGS_NAME)
                data_valids = data_flags == measurements.wod.po4.constants.DATA_VALID_FLAG

                data_profile_flag = self._load_value_from_file(file, measurements.wod.po4.constants.DATA_PROFILE_FLAG_NAME)
                data_profile_valid = data_profile_flag == measurements.wod.po4.constants.DATA_PROFILE_VALID_FLAG

                valid_mask = np.logical_and(z_valids, data_valids) * data_profile_valid
                valid_mask = np.logical_and(valid_mask, data != measurements.wod.po4.constants.MISSING_DATA_VALUE)

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
