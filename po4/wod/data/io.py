import numpy as np
import logging
logger = logging.getLogger(__name__)

from .cruise import Cruise_Collection
from measurements.po4.wod.data.results import Measurements_Sorted, Measurements_Unsorted

from .constants import DATA_DIR, CRUISES_FILE, MEASUREMENTS_DICT_SORTED_FILE, MEASUREMENTS_DICT_UNSORTED_FILE, MEASUREMENTS_POINTS_FILE, MEASUREMENTS_RESULTS_FILE


def save_cruises(data_dir=DATA_DIR, cruise_file=CRUISES_FILE):
    cc = Cruise_Collection()
    cc.load_cruises_from_netcdf(data_dir)
    cc.save_cruises_to_pickle_file(cruise_file)

def save_measurement_dict_sorted(cruise_file=CRUISES_FILE, dict_file=MEASUREMENTS_DICT_SORTED_FILE):
    cc = Cruise_Collection()
    cc.load_cruises_from_pickle_file(cruise_file)
    measurements = Measurements_Sorted()
    measurements.add_cruises(cc.cruises)
    measurements.save(dict_file)

def load_measurement_dict_sorted(dict_file=MEASUREMENTS_DICT_UNSORTED_FILE):
    measurements = Measurements_Sorted()
    measurements.load(dict_file)
    return measurements
    
def save_measurement_dict_unsorted(cruise_file=CRUISES_FILE, dict_file=MEASUREMENTS_DICT_UNSORTED_FILE):
    cc = Cruise_Collection()
    cc.load_cruises_from_pickle_file(cruise_file)
    measurements = Measurements_Unsorted()
    measurements.add_cruises(cc.cruises)
    measurements.save(dict_file)

def load_measurement_dict_unsorted(dict_file=MEASUREMENTS_DICT_UNSORTED_FILE):
    measurements = Measurements_Unsorted()
    measurements.load(dict_file)
    return measurements

def save_measurements(dict_file=MEASUREMENTS_DICT_UNSORTED_FILE, points_file=MEASUREMENTS_POINTS_FILE, results_file=MEASUREMENTS_RESULTS_FILE):
    
    logger.debug('Loading and calculating measurements.')
    
    ## load measurements
    measurements = Measurements_Unsorted()
    measurements.load(dict_file)
    
    values = measurements.means()
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
    
    ## saving points and values
    logger.debug('Saving %d measurement points to %s.' % (len(values), points_file))
    np.save(points_file, points)
    logger.debug('Saving %d measurement results to %s.' % (len(values), results_file))
    np.save(results_file, results)
    
    logger.debug('Measurements saved.')


def load_measurement_points(file=MEASUREMENTS_POINTS_FILE):
    logger.debug('Loading measurement points from %s.' % file)
    values = np.load(file)
    return values

def load_measurement_results(file=MEASUREMENTS_RESULTS_FILE):
    logger.debug('Loading measurement results from %s.' % file)
    values = np.load(file)
    return values