import calendar
import datetime
import numpy as np
import logging
logger = logging.getLogger(__name__)

import util.io
import measurements.util.data


def load_data_from_file(data_file, calculation_funtion):
    ## try to load data
    try:
        data = np.load(data_file)
        logger.debug('{} DOP data sets loaded from {}.'.format(data.shape[0], data_file))
    
    ## otherwise calculate data and save
    except (OSError, IOError):
        data = calculation_funtion()
        logger.debug('{} DOP data sets saved to {}.'.format(data.shape[0], data_file))
        util.io.save_npy_and_txt(data, data_file, make_read_only=True)
    
    assert data.ndim == 2
    assert data.shape[1] == 5
    
    return data




def prepare_ladolfi_data(measurement_file, time_start, time_end, valid_data_flag):
    
    # convert missing values to - Inf
    def convert_missing_values_to_nan(value_string):
        try:
            value = float(value_string)
        except ValueError:
            value = - float('nan')
        
        return value
    
    # load data, columns: lat, long, depth, dop, flag
    data = np.loadtxt(measurement_file, converters={3: convert_missing_values_to_nan})
    if data.shape[1] != 5:
        raise ValueError('The data in {} has not 5 columns. Its shape is {}'.format(measurement_file, data.shape))
    
    # skip invalid data and flag
    data = data[data[:,4]==valid_data_flag]
    data = data[:,:4]
    data = data[data[:,3]>=0]
    data = data[np.logical_not(np.isnan(data[:,3]))]
    
    # move columns in order long, lat, depth, dop
    data[:,[0, 1]] = data[:,[1, 0]]
    
    # sort data in priority by long, lat, depth
    data = data[np.lexsort((data[:,2], data[:,1], data[:,0]))]
    
    # interpolate time
    n = data.shape[0]
    t = np.arange(n) / (n-1) * (time_end - time_start) + time_start
    t = t.reshape([n, 1])
    data = np.concatenate((t, data), axis=1)
    
    logger.debug('{} DOP data sets loaded from {}.'.format(data.shape[0], measurement_file))
    
    return data


def load_ladolfi_2002():
    from .constants import LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_TIME_START, LADOLFI_2002_TIME_END, LADOLFI_2002_VALID_DATA_FLAG, LADOLFI_2002_DATA_FILE
    
    data = load_data_from_file(LADOLFI_2002_DATA_FILE, lambda : prepare_ladolfi_data(LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_TIME_START, LADOLFI_2002_TIME_END, LADOLFI_2002_VALID_DATA_FLAG))
    
    return data


def load_ladolfi_2004():
    from .constants import LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_TIME_START, LADOLFI_2004_TIME_END, LADOLFI_2004_VALID_DATA_FLAG, LADOLFI_2004_DATA_FILE
    
    data = load_data_from_file(LADOLFI_2004_DATA_FILE, lambda : prepare_ladolfi_data(LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_TIME_START, LADOLFI_2004_TIME_END, LADOLFI_2004_VALID_DATA_FLAG))
    
    return data




def prepare_yoshimura_data(measurement_file):
    ## columns: lat, long, date, time, dop
    
    ## calculate time
    def convert_date_to_number_of_days_in_year(value_bytes):
        value_string = value_bytes.decode()
        d = datetime.datetime.strptime(value_string, '%d-%B-%Y').date()
        number_of_days = d.timetuple().tm_yday
        logger.debug('Convering: "{}" is the {}. day in a year.'.format(value_bytes, number_of_days))
        return number_of_days
    
    def convert_date_to_number_of_all_days_in_year(value_bytes):
        value_string = value_bytes.decode()
        d = datetime.datetime.strptime(value_string, '%d-%B-%Y').date()
        number_of_days_in_year = calendar.isleap(d.year) + 365
        logger.debug('Convering: The year of "{}" has {} days.'.format(value_bytes, number_of_days_in_year))
        return number_of_days_in_year
    
    def convert_time_to_day_frac(value_bytes):
        value_string = value_bytes.decode()
        dt = datetime.datetime.strptime(value_string, '%H:%M')
        hour_frac = dt.hour + dt.minute / 60
        day_frac = hour_frac / 24
        logger.debug('Convering: The time "{}" is {} of a day.'.format(value_bytes, day_frac))
        return day_frac
    
    number_of_days_in_year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_number_of_days_in_year})
    number_of_all_days_in_year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_number_of_all_days_in_year})
    day_frac = np.loadtxt(measurement_file, usecols=(3,), converters={3: convert_time_to_day_frac})
    year_frac = (number_of_days_in_year + day_frac) / number_of_all_days_in_year
    
    
    
    ## load lat, long and dop
    lat = np.loadtxt(measurement_file, usecols=(0,))
    long = np.loadtxt(measurement_file, usecols=(1,))
    dop = np.loadtxt(measurement_file, usecols=(4,))
    depth = dop * 0
    
    ## concatenate columns in order long, lat, depth, dop
    n = len(lat)
    data = np.concatenate((year_frac.reshape([n, 1]), long.reshape([n, 1]), lat.reshape([n, 1]), depth.reshape([n, 1]), dop.reshape([n, 1])), axis=1)
    
    logger.debug('{} DOP data sets loaded from {}.'.format(data.shape[0], measurement_file))
    
    return data



def load_yoshimura_2007():
    from .constants import YOSHIMURA_2007_DATA_FILE, YOSHIMURA_2007_MEASUREMENT_FILE
    
    data = load_data_from_file(YOSHIMURA_2007_DATA_FILE, lambda : prepare_yoshimura_data(YOSHIMURA_2007_MEASUREMENT_FILE))
    
    return data



def load_data():
    data = np.concatenate((load_ladolfi_2002(), load_ladolfi_2004(), load_yoshimura_2007()))
    return data

def load_points_and_values():
    data = load_data()
    points = data[:, :-1]
    values = data[:, -1]
    return (points, values)

def load_as_measurements():
    (points, values) = load_points_and_values()
    measurement_data = measurements.util.data.Measurements_Unsorted()
    measurement_data.add_results(points, values)
    return measurement_data