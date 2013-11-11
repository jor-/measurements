import argparse
import os.path
import logging

from measurements.po4.wod.data.cruise import Cruise_Collection
from measurements.po4.wod.data.measurements import Measurements

parser = argparse.ArgumentParser(description='Read netcdf cruise files, extract measurements and save them.')
parser.add_argument('-p', '--path', default='/work_j2/sunip229/tmp/', help='Path where to save the files.')
parser.add_argument('-d', '--debug', action='store_true', help='Enable output of debug informations.')
parser.add_argument('--version', action='version', version='%(prog)s 0.1')
args = vars(parser.parse_args())

path = args['path']
debug = args['debug']
if debug:
    logging.basicConfig(level=logging.DEBUG)

cruise_file = os.path.join(path, 'cruises_list.py')
measurements_dict_file = os.path.join(path, 'measurements_dict_by_coordinates.py')

cc = Cruise_Collection()
cc.load_cruises_from_pickle_file(cruise_file)
measurements = Measurements()
measurements.add_cruises_with_coordinates(cc.cruises)
measurements.save(measurements_dict_file)