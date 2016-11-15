import overrides
import numpy as np

import measurements.universal.data
import measurements.universal.constants
import measurements.constants
import measurements.po4.constants
import measurements.po4.wod.cruise
import measurements.po4.wod.dict
import measurements.po4.wod.constants

import util.math.sort
import util.cache.file
import util.cache.memory
import util.logging

logger = util.logging.logger



## data load functions

@util.cache.file.decorator(cache_file_function=lambda :measurements.po4.wod.constants.CRUISES_LIST_FILE)
def cruises_list():
    cc = measurements.po4.wod.cruise.CruiseCollection()
    cc.load_cruises_from_netcdf_files(measurements.po4.wod.constants.CRUISES_DATA_DIR)
    return cc.cruises


@util.cache.file.decorator(cache_file_function=lambda :measurements.po4.wod.constants.MEASUREMENTS_DICT_UNSORTED_FILE)
def measurement_dict():
    m = measurements.po4.wod.dict.MeasurementsUnsortedDict()
    m.add_cruises(cruises_list())
    return m


@util.cache.file.decorator(cache_file_function=lambda :measurements.po4.wod.constants.POINTS_AND_RESULTS_FILE)
def points_and_results():
    logger.debug('Loading and calculating measurements.')

    ## load measurements
    m = measurement_dict()

    values = m.items()
    assert values.ndim == 2
    n = values.shape[1]
    assert n == 5

    ## sort measurements
    sorted_indices = util.math.sort.lex_sorted_indices(values)
    assert sorted_indices.ndim == 1
    values = values[sorted_indices]

    ## split measurements
    points = values[:, :-1]
    results = values[:, -1]

    return{'points': points, 'results': results}


def points():
    points_and_results()['points']

def results():
    points_and_results()['results']



## measurement classes

class Measurements(measurements.universal.data.MeasurementsAnnualPeriodicCache):
    
    def __init__(self, sample_t_dim=measurements.po4.wod.constants.SAMPLE_T_DIM, min_measurements_correlation=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        
        tracer = 'po4'        
        data_set_name = 'wod_2013'        
        
        sample_lsm = measurements.po4.wod.constants.SAMPLE_LSM
        sample_lsm.t_dim = sample_t_dim
        min_deviation = measurements.po4.constants.DEVIATION_MIN_VALUE
        
        super().__init__(sample_lsm, tracer=tracer, data_set_name=data_set_name, min_standard_deviation=min_deviation, min_measurements_correlation=min_measurements_correlation)
        
        self.fill_strategy = 'interpolate'
        
        try:
            INTERPOLATOR_OPTIONS = measurements.po4.wod.constants.INTERPOLATOR_OPTIONS
        except AttributeError:
            pass
        else:
            try:
                interpolator_option = INTERPOLATOR_OPTIONS['mean']['concentration'][measurements.constants.MEAN_MIN_MEASUREMENTS][str(sample_lsm)]
            except KeyError:
                pass
            else:
                self.set_interpolator_options('concentration_means', interpolator_option)
            try:
                interpolator_option = INTERPOLATOR_OPTIONS['deviation']['concentration'][measurements.constants.DEVIATION_MIN_MEASUREMENTS][str(sample_lsm)]
            except KeyError:
                pass
            else:
                self.set_interpolator_options('concentration_standard_deviations', interpolator_option)
            try:
                interpolator_option = INTERPOLATOR_OPTIONS['deviation']['average_noise'][measurements.constants.DEVIATION_MIN_MEASUREMENTS][str(sample_lsm)]
            except KeyError:
                pass
            else:
                self.set_interpolator_options('average_noise_standard_deviations', interpolator_option)

    
    @property
    @util.cache.memory.method_decorator()
    @util.cache.file.decorator()
    @overrides.overrides
    def points(self):
        return points()

    @property
    @util.cache.memory.method_decorator()
    @util.cache.file.decorator()
    @overrides.overrides
    def values(self):
        return results()



class MeasurementsNearWater(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):
    
    def __init__(self, water_lsm=None, max_box_distance_to_water=0, sample_t_dim=measurements.po4.wod.constants.SAMPLE_T_DIM, min_measurements_correlation=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurements = Measurements(sample_t_dim=sample_t_dim, min_measurements_correlation=min_measurements_correlation)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)

