import numpy as np

import util.logging

import measurements.wod.cruise
import measurements.wod.plankton.constants


class Cruise(measurements.wod.cruise.Cruise):

    PLANKTON_GROUP_CODE = None

    def _load_depths_and_values(self, file):
        depths = []
        data = []

        # check each variable
        for name, variable in file.variables.items():
            # check if data record
            if name.startswith(measurements.wod.plankton.constants.DATA_NAME_PREFIX):
                # check wanted plankton
                try:
                    plankton_group_code = getattr(variable, measurements.wod.plankton.constants.DATA_PLANKTON_GROUPE_CODE_NAME)
                except AttributeError:
                    util.logging.warn('Value {} missing for variable {}.'.format(measurements.wod.plankton.constants.DATA_PLANKTON_GROUPE_CODE_NAME, name))
                else:
                    if plankton_group_code == self.PLANKTON_GROUP_CODE:
                        # check quality flag
                        try:
                            flag_value = getattr(variable, measurements.wod.plankton.constants.DATA_FLAG_NAME)
                        except AttributeError:
                            util.logging.warn('Value {} missing for variable {}.'.format(measurements.wod.plankton.constants.DATA_FLAG_NAME, name))
                        else:
                            if flag_value == measurements.wod.plankton.constants.DATA_VALID_FLAG:
                                # check unit and data format
                                unit = getattr(variable, measurements.wod.plankton.constants.DATA_UNIT_NAME)
                                assert unit == measurements.wod.plankton.constants.DATA_UNIT
                                comment = getattr(variable, measurements.wod.plankton.constants.DATA_COMMENT_NAME)
                                assert comment == measurements.wod.plankton.constants.DATA_COMMENT

                                # add data
                                values = variable.data
                                assert len(values) == 2
                                value = float(values[1])
                                assert value >= 0
                                data.append(value)

                                # add depth
                                upper_depth = getattr(variable, measurements.wod.plankton.constants.DATA_UPPER_DEPTH_NAME)
                                lower_depth = getattr(variable, measurements.wod.plankton.constants.DATA_LOWER_DEPTH_NAME)
                                assert 0 <= upper_depth <= lower_depth
                                depth = (upper_depth + lower_depth) / 2
                                depths.append(depth)
                                util.logging.debug('A  valid measurement found.')
                            else:
                                util.logging.debug('An invalid measurement found (flag {}).'.format(flag_value))

        depths = np.array(depths)
        data = np.array(data)

        return depths, data


class CruisePhytoplankton(Cruise):

    PLANKTON_GROUP_CODE = measurements.wod.plankton.constants.PHYTOPLANKTON_GROUP_CODE


class CruiseZooplankton(Cruise):

    PLANKTON_GROUP_CODE = measurements.wod.plankton.constants.ZOOPLANKTON_GROUP_CODE
