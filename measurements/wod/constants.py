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


# data set names

WOD_13_DATA_SET_NAME = 'wod_2013'
WOD_18_DATA_SET_NAME = 'wod_2018'


# data dirs and files

DATA_DIR = os.path.join(measurements.constants.BASE_DIR, '{tracer}', '{data_set_name}', 'data')
CRUISES_FILE = os.path.join(DATA_DIR, 'raw', 'cruises.tar.bz2')
CRUISES_COLLECTION_FILE = os.path.join(DATA_DIR, 'cruises_collection.ppy')
MEASUREMENTS_DICT_FILE = os.path.join(DATA_DIR, 'measurement_dict.ppy')
POINTS_AND_RESULTS_FILE = os.path.join(DATA_DIR, 'points_and_results.npz')
