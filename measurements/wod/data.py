import overrides

import util.cache.file
import util.io.np
import util.io.object
import util.math.sort

import measurements.universal.data
import measurements.wod.constants
import measurements.wod.cruise
import measurements.wod.dict
import measurements.wod.plankton.cruise
import measurements.wod.plankton.constants
import measurements.wod.po4.cruise
import measurements.wod.po4.constants

SUPPORTED_TRACERS = (measurements.wod.po4.constants.TRACER,) + measurements.wod.plankton.constants.TRACERS


# *** cruise collection *** #

def cruises_collection_calculate(tracer):
    # init cruise collection
    if tracer == 'po4':
        cruises_class = measurements.wod.po4.cruise.CruisePO4
    elif tracer == 'zooplankton':
        cruises_class = measurements.wod.plankton.cruise.CruiseZooplankton
    elif tracer == 'phytoplankton':
        cruises_class = measurements.wod.plankton.cruise.CruisePhytoplankton
    else:
        raise ValueError('Tracer {} is not supported. Only {} are supported.'.format(tracer, SUPPORTED_TRACERS))
    cruises_collection = measurements.wod.cruise.CruisesCollection(cruises_class)
    # load cruises
    cruises_file = measurements.wod.constants.CRUISES_FILE.format(tracer=tracer)
    cruises_collection.add_cruises_from_netcdf_files_in_tar_file(cruises_file)
    # return cruise collection
    return cruises_collection


def cruises_collection_file(tracer):
    return measurements.wod.constants.CRUISES_COLLECTION_FILE.format(tracer=tracer)


@util.cache.file.decorator(cache_file_function=cruises_collection_file, load_function=util.io.object.load, save_function=util.io.object.save)
def cruises_collection(tracer):
    return cruises_collection_calculate(tracer)


# *** measurement dict *** #

def measurements_dict_calculate(tracer):
    cruises_collection_tracer = cruises_collection(tracer)
    m = measurements.wod.dict.MeasurementsDict()
    m.add_cruises(cruises_collection_tracer)
    return m


def measurements_dict_file(tracer):
    return measurements.wod.constants.MEASUREMENTS_DICT_FILE.format(tracer=tracer)


@util.cache.file.decorator(cache_file_function=measurements_dict_file, load_function=measurements.wod.dict.load, save_function=measurements.wod.dict.save)
def measurements_dict(tracer):
    return measurements_dict_calculate(tracer)


# *** points and results *** #

def points_and_results_calculate(tracer):
    util.logging.debug('Loading and calculating measurements.')

    # load measurements
    m = measurements_dict(tracer)

    values = m.items()
    assert values.ndim == 2

    if len(values) > 0:
        assert values.shape[1] == 5
        # sort measurements
        sorted_indices = util.math.sort.lex_sorted_indices(values)
        assert sorted_indices.ndim == 1
        values = values[sorted_indices]
    else:
        values = values.reshape(0, 5)

    # return
    return values


def points_and_results_file(tracer):
    return measurements.wod.constants.POINTS_AND_RESULTS_FILE.format(tracer=tracer)


@util.cache.file.decorator(cache_file_function=points_and_results_file, load_function=util.io.np.load, save_function=util.io.np.save)
def points_and_results(tracer):
    return points_and_results_calculate(tracer)


def points(tracer):
    return points_and_results(tracer)[:, :-1]


def results(tracer):
    return points_and_results(tracer)[:, -1]


# ***  measurement classes *** #

class MeasurementsBase(measurements.universal.data.MeasurementsAnnualPeriodicCache):

    def __init__(self, tracer, sample_lsm, min_standard_deviation, min_measurements_correlation):
        data_set_name = measurements.wod.constants.DATA_SET_NAME
        super().__init__(tracer=tracer, data_set_name=data_set_name, sample_lsm=sample_lsm, min_standard_deviation=min_standard_deviation, min_measurements_correlation=min_measurements_correlation)

    @property
    @overrides.overrides
    def points(self):
        return points(self.tracer)

    @property
    @overrides.overrides
    def values(self):
        return results(self.tracer)


class Measurements(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):

    def __init__(self, tracer, sample_lsm, min_standard_deviation, min_measurements_correlation, water_lsm=None, max_box_distance_to_water=None):
        measurements = MeasurementsBase(tracer, sample_lsm, min_standard_deviation, min_measurements_correlation)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)
