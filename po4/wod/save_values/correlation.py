import argparse
import os.path

from measurements.po4.wod.data.results import Measurements_Unsorted as M
from measurements.po4.wod.correlation.constants import EQUAL_BOUNDS

import util.logging

PATH = '/work_O2/sunip229/NDOP/measurements/po4/wod13/analysis/correlation'

if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser(description='Save correlation of WOD data.')
    parser.add_argument('-m', '--min', type=int)
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    args = vars(parser.parse_args())
    
    with util.logging.Logger(disp_stdout=args['debug']):
        min_measurements = args['min']
    
        m = M()
        m.load()
        
        ms = m.filter_same_indices(equal_bounds=EQUAL_BOUNDS, discard_year=True, only_one_per_year=True, min_measurements=min_measurements)
        ms.save(os.path.join(PATH, '{}_min_measurements_-_filter_same_indices.ppy'.format(min_measurements)))
        
        correlation = m.correlation(EQUAL_BOUNDS, min_measurements=min_measurements, measurements_same_indices=ms)
        correlation.save(os.path.join(PATH, '{}_min_measurements_-_correlation.ppy'.format(min_measurements)))