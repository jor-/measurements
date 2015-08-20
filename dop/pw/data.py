import calendar
import datetime
import numpy as np

import measurements.dop.pw.constants
import measurements.util.data

import util.math.sort
import util.cache
import util.logging
logger = util.logging.logger



def prepare_ladolfi_data(measurement_file, start_date, end_date, valid_data_flag):

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
    def convert_date_to_year_float(date):
        number_of_days_in_year = date.timetuple().tm_yday
        number_of_total_days_in_year = calendar.isleap(date.year) + 365
        year_frac = number_of_days_in_year / number_of_total_days_in_year
        year = date.year
        time = year + year_frac
        return time

    start_year_float = convert_date_to_year_float(start_date)
    end_year_float = convert_date_to_year_float(end_date)

    n = data.shape[0]
    t = np.arange(n) / (n-1) * (end_year_float - start_year_float) + start_year_float
    t = t.reshape([n, 1])
    data = np.concatenate((t, data), axis=1)

    logger.debug('{} DOP data sets loaded from {}.'.format(data.shape[0], measurement_file))

    return data



def load_ladolfi_2002():
    from measurements.dop.pw.constants import DATA_FILENAME, LADOLFI_2002_DIR, LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_START_DATE, LADOLFI_2002_END_DATE, LADOLFI_2002_VALID_DATA_FLAG

    cache = util.cache.HDD_NPY_Cache(LADOLFI_2002_DIR)
    calculation_function = lambda : prepare_ladolfi_data(LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_START_DATE, LADOLFI_2002_END_DATE, LADOLFI_2002_VALID_DATA_FLAG)

    return cache.get_value(DATA_FILENAME, calculation_function)




def load_ladolfi_2004():
    from measurements.dop.pw.constants import DATA_FILENAME, LADOLFI_2004_DIR, LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_START_DATE, LADOLFI_2004_END_DATE, LADOLFI_2004_VALID_DATA_FLAG

    cache = util.cache.HDD_NPY_Cache(LADOLFI_2004_DIR)
    calculation_function = lambda : prepare_ladolfi_data(LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_START_DATE, LADOLFI_2004_END_DATE, LADOLFI_2004_VALID_DATA_FLAG)

    return cache.get_value(DATA_FILENAME, calculation_function)




def prepare_yoshimura_data(measurement_file):
    ## columns: lat, long, date, time, dop

    ## calculate time
    def convert_date_to_year(value_bytes):
        value_string = value_bytes.decode()
        d = datetime.datetime.strptime(value_string, '%d-%B-%Y').date()
        year = d.year
        logger.debug('Convering: "{}" is in the year {}.'.format(value_bytes, year))
        return year

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

    year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_year})

    number_of_days_in_year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_number_of_days_in_year})
    number_of_all_days_in_year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_number_of_all_days_in_year})
    day_frac = np.loadtxt(measurement_file, usecols=(3,), converters={3: convert_time_to_day_frac})
    year_frac = (number_of_days_in_year + day_frac) / number_of_all_days_in_year

    time = year + year_frac


    ## load lat, long and dop
    lat = np.loadtxt(measurement_file, usecols=(0,))
    long = np.loadtxt(measurement_file, usecols=(1,))
    dop = np.loadtxt(measurement_file, usecols=(4,))
    depth = dop * 0


    ## concatenate columns in order long, lat, depth, dop
    data = np.concatenate((time[:,np.newaxis], long[:,np.newaxis], lat[:,np.newaxis], depth[:,np.newaxis], dop[:,np.newaxis]), axis=1)

    logger.debug('{} DOP data sets loaded from {}.'.format(data.shape[0], measurement_file))

    return data




def load_yoshimura_2007():
    from measurements.dop.pw.constants import DATA_FILENAME, YOSHIMURA_2007_DIR, YOSHIMURA_2007_MEASUREMENT_FILE

    cache = util.cache.HDD_NPY_Cache(YOSHIMURA_2007_DIR)
    calculation_function = lambda : prepare_yoshimura_data(YOSHIMURA_2007_MEASUREMENT_FILE)

    return cache.get_value(DATA_FILENAME, calculation_function)



def data():
    data = np.concatenate((load_ladolfi_2002(), load_ladolfi_2004(), load_yoshimura_2007()))
    return data

def points_and_values():
    data_array = data()
    points = data_array[:, :-1]
    values = data_array[:, -1]
    return (points, values)


def data_calculate():
    values = np.concatenate((load_ladolfi_2002(), load_ladolfi_2004(), load_yoshimura_2007()))

    ## sort measurements
    sorted_indices = util.math.sort.lex_sorted_indices(values)
    assert sorted_indices.ndim == 1
    values = values[sorted_indices]

    ## split measurements
    points = values[:, :-1]
    results = values[:, -1]

    return (points, results)


def points():
    cache = util.cache.HDD_NPY_Cache(measurements.dop.pw.constants.DATA_DIR)
    return cache.get_value(measurements.dop.pw.constants.MEASUREMENTS_POINTS_FILENAME, lambda :measurements_calculate()[0])

def results():
    cache = util.cache.HDD_NPY_Cache(measurements.dop.pw.constants.DATA_DIR)
    return cache.get_value(measurements.dop.pw.constants.MEASUREMENTS_RESULTS_FILENAME, lambda :measurements_calculate()[1])


def points_and_results():
    return (points(), results())


def measurement_dict():
    (points, values) = points_and_values()
    measurement_data = measurements.util.data.Measurements()
    measurement_data.append_values(points, values)
    return measurement_data