import numpy as np


import measurements.po4.wod.data.cruise
import measurements.po4.wod.data.results
import measurements.po4.wod.data.constants

import util.cache

import util.logging
logger = util.logging.get_logger()



def cruises_list_calculate():
    cc = measurements.po4.wod.data.cruise.Cruise_Collection()
    cc.load_cruises_from_netcdf(measurements.po4.wod.data.constants.CRUISES_DATA_DIR)
    return cc.cruises

def cruises_list():
    cache = util.cache.HDD_Pickle_Cache(measurements.po4.wod.data.constants.DATA_DIR)
    return cache.get_value(measurements.po4.wod.data.constants.CRUISES_LIST_FILENAME, cruises_list_calculate)


def measurement_dict_sorted_calculate():
    m = measurements.po4.wod.data.results.Measurements_Sorted()
    m.add_cruises(cruises_list())
    return m

def measurement_dict_sorted():
    cache = util.cache.HDD_Object_Cache(measurements.po4.wod.data.constants.DATA_DIR, measurements.po4.wod.data.results.Measurements_Sorted())
    return cache.get_value(measurements.po4.wod.data.constants.MEASUREMENTS_DICT_SORTED_FILENAME, measurement_dict_sorted_calculate)


def measurement_dict_unsorted_calculate():
    m = measurements.po4.wod.data.results.Measurements_Unorted()
    m.add_cruises(cruises_list())
    return m

def measurement_dict_unsorted():
    cache = util.cache.HDD_Object_Cache(measurements.po4.wod.data.constants.DATA_DIR, measurements.po4.wod.data.results.Measurements_Unsorted())
    return cache.get_value(measurements.po4.wod.data.constants.MEASUREMENTS_DICT_UNSORTED_FILENAME, measurement_dict_unsorted_calculate)


def measurement_dict():
    return measurement_dict_unsorted()
    

def measurements_calculate():
    logger.debug('Loading and calculating measurements.')
    
    ## load measurements
    m = measurement_dict_unsorted()
    
    values = m.all_points_and_results()
    assert values.ndim == 2
    n = values.shape[1]
    assert n == 5
    
    ## sort measurements
    lex_list = []
    for i in range(n-1, -1, -1):
        lex_list.append(values[:, i])
    sorted_indices = np.lexsort(lex_list)
    value = values[sorted_indices]
    
    ## split measurements
    points = values[:, :-1]
    results = values[:, -1]
    
    return (points, results)

def measurement_points():
    cache = util.cache.HDD_NPY_Cache(measurements.po4.wod.data.constants.DATA_DIR)
    return cache.get_value(measurements.po4.wod.data.constants.MEASUREMENTS_POINTS_FILENAME, lambda :measurements_calculate()[0])

def measurement_results():
    cache = util.cache.HDD_NPY_Cache(measurements.po4.wod.data.constants.DATA_DIR)
    return cache.get_value(measurements.po4.wod.data.constants.MEASUREMENTS_RESULTS_FILENAME, lambda :measurements_calculate()[1])

def measurements_points_and_results():
    return (measurement_points(), measurement_results())
