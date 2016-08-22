import numpy as np

import util.math.sort
import util.cache.file_based
import util.logging

import measurements.po4.wod.cruise
import measurements.po4.wod.dict
import measurements.po4.wod.constants

logger = util.logging.logger



def cruises_list_cache_file():
    return measurements.po4.wod.constants.CRUISES_LIST_FILE

@util.cache.file_based.decorator(cache_file_function=cruises_list_cache_file)
def cruises_list():
    cc = measurements.po4.wod.cruise.CruiseCollection()
    cc.load_cruises_from_netcdf(measurements.po4.wod.constants.CRUISES_DATA_DIR)
    return cc.cruises


def measurement_dict_cache_file():
    return measurements.po4.wod.constants.MEASUREMENTS_DICT_UNSORTED_FILE

@util.cache.file_based.decorator(cache_file_function=measurement_dict_cache_file)
def  measurement_dict():
    m = measurements.po4.wod.dict.MeasurementsUnsortedDict()
    m.add_cruises(cruises_list())
    return m


def points_and_results_cache_file():
    return measurements.po4.wod.constants.POINTS_AND_RESULTS_FILE

@util.cache.file_based.decorator(cache_file_function=points_and_results_cache_file)
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

