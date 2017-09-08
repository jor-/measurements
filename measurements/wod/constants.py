import datetime
import os.path

import measurements.constants


# informations for the wod cruises

DATE_VARIABLE = {
    'name': 'date',
    'unit': b'YYYYMMDD',
    'unit_name': 'comment'}

TIME_VARIABLE = {
    'name': 'GMT_time',
    'unit': None,
    'unit_name': None}

START_TIME_VARIABLE = {
    'name': 'GMT_sample_start_time',
    'unit': b'hour',
    'unit_name': 'units'}

DATE_TIME_VARIABLE = {
    'name': 'time',
    'unit': b'days since 1770-01-01 00:00:00',
    'unit_name': 'units',
    'base': datetime.datetime(1770, 1, 1)}

LATITUDE_VARIABLE = {
    'name': 'lat',
    'unit': b'degrees_north',
    'unit_name': 'units'}

LONGITUDE_VARIABLE = {
    'name': 'lon',
    'unit': b'degrees_east',
    'unit_name': 'units'}

DATE_NAME = 'date'
DATE_UNIT = b'YYYYMMDD'
DATE_UNIT_NAME = 'comment'

TIME_NAME = 'GMT_time'
TIME_UNIT = None
TIME_UNIT_NAME = None

START_TIME_NAME = 'GMT_sample_start_time'
START_TIME_UNIT = b'hour'
START_TIME_UNIT_NAME = 'units'

DATE_TIME_NAME = 'time'
DATE_TIME_UNIT = b'days since 1770-01-01 00:00:00'
DATE_TIME_UNIT_NAME = 'units'
DATE_TIME_BASE = datetime.datetime(1770, 1, 1)

LATITUDE_NAME = 'lat'
LATITUDE_UNIT = b'degrees_north'
LATITUDE_UNIT_NAME = 'units'

LONGITUDE_NAME = 'lon'
LONGITUDE_UNIT = b'degrees_east'
LONGITUDE_UNIT_NAME = 'units'


# general informations

DATA_SET_NAME = 'wod_2013'


# data dirs and files

DATA_DIR = os.path.join(measurements.constants.BASE_DIR, '{tracer}', DATA_SET_NAME, 'data')
CRUISES_FILE = os.path.join(DATA_DIR, 'raw', 'cruises.tar')
CRUISES_COLLECTION_FILE = os.path.join(DATA_DIR, 'cruises_collection.ppy')
MEASUREMENTS_DICT_FILE = os.path.join(DATA_DIR, 'measurement_dict.ppy')
POINTS_AND_RESULTS_FILE = os.path.join(DATA_DIR, 'points_and_results.npz')
