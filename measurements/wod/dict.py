import util.logging

import measurements.universal.dict


class MeasurementsDict(measurements.universal.dict.MeasurementsDict):

    def add_cruises(self, cruises):
        util.logging.debug('Adding {} cruises to measurements dict.'.format(len(cruises)))
        for cruise in cruises:
            t = cruise.time
            x = cruise.longitude
            y = cruise.latitude
            z = cruise.depths
            results = cruise.values.astype(float)
            assert len(z) == len(results)
            for z_i, result_i in zip(z, results):
                index = (t, x, y, z_i)
                self.append_value(index, result_i)


def save(file, measurements_dict):
    measurements_dict.save(file)


def load(file):
    return MeasurementsDict.load(file)
