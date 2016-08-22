import os.path

import measurements.universal.dict
import measurements.po4.wod.constants


class MeasurementsDict(measurements.universal.dict.MeasurementsDict):

    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)

    def add_cruises(self, cruises):
        measurements_dict = self.measurements_dict

        ## insert results in dict
        for cruise in cruises:
            x = cruise.x
            y = cruise.y
            z = cruise.z
            t = cruise.dt_float
            results = cruise.po4.astype(float)

            for i in range(results.size):
                index = (t, x, y, z[i])
                self.add_result(index, results[i])


    def save(self, file=measurements.po4.wod.constants.MEASUREMENTS_DICT_UNSORTED_FILE):
        super().save(file)

    @classmethod
    def load(cls, file=measurements.po4.wod.constants.MEASUREMENTS_DICT_UNSORTED_FILE):
        return super().load(file)



class MeasurementsUnsortedDict(MeasurementsDict):

    def __init__(self):
        super().__init__(sorted=False)



class MeasurementsSortedDict(MeasurementsDict):

    def __init__(self):
        super().__init__(sorted=False)

    def save(self, file=measurements.po4.wod.constants.MEASUREMENTS_DICT_SORTED_FILE):
        super().save(file)

    @classmethod
    def load(cls, file=measurements.po4.wod.constants.MEASUREMENTS_DICT_SORTED_FILE):
        return super().load(file)

