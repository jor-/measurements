import calendar
import datetime
import os.path

import numpy as np
import overrides

import util.math.sort
import util.cache.file_based
import util.cache.memory_based
import util.logging

import measurements.universal.data
import measurements.universal.constants
import measurements.constants
import measurements.dop.constants
import measurements.dop.pw.constants

logger = util.logging.logger


## data load functions

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


@util.cache.file_based.decorator(cache_file_function=lambda: os.path.join(measurements.dop.pw.constants.LADOLFI_2002_DIR, measurements.dop.pw.constants.DATA_FILENAME))
def load_ladolfi_2002():
    from measurements.dop.pw.constants import LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_START_DATE, LADOLFI_2002_END_DATE, LADOLFI_2002_VALID_DATA_FLAG
    return prepare_ladolfi_data(LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_START_DATE, LADOLFI_2002_END_DATE, LADOLFI_2002_VALID_DATA_FLAG)


@util.cache.file_based.decorator(cache_file_function=lambda: os.path.join(measurements.dop.pw.constants.LADOLFI_2004_DIR, measurements.dop.pw.constants.DATA_FILENAME))
def load_ladolfi_2004():
    from measurements.dop.pw.constants import LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_START_DATE, LADOLFI_2004_END_DATE, LADOLFI_2004_VALID_DATA_FLAG
    return prepare_ladolfi_data(LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_START_DATE, LADOLFI_2004_END_DATE, LADOLFI_2004_VALID_DATA_FLAG)



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


@util.cache.file_based.decorator(cache_file_function=lambda: os.path.join(measurements.dop.pw.constants.YOSHIMURA_2007_DIR, measurements.dop.pw.constants.DATA_FILENAME))
def load_yoshimura_2007():
    from measurements.dop.pw.constants import YOSHIMURA_2007_MEASUREMENT_FILE
    return prepare_yoshimura_data(YOSHIMURA_2007_MEASUREMENT_FILE)



## measurement classes

class MeasurementsSingleBase(measurements.universal.data.MeasurementsAnnualPeriodicCache):
    
    def __init__(self, data_set_name, load_data_function, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        
        tracer = 'dop'        
        data_set_name = data_set_name       
        
        sample_lsm = measurements.dop.pw.constants.SAMPLE_LSM
        min_standard_deviation = measurements.dop.constants.DEVIATION_MIN_VALUE
        
        super().__init__(sample_lsm, tracer=tracer, data_set_name=data_set_name, min_standard_deviation=min_standard_deviation, min_measurements_correlations=min_measurements_correlations)
        
        self._load_data_function = load_data_function
    
    
    def __str__(self):
        string = super().__str__()
        if self.min_measurements_correlations < float('inf'):
            string = string + '({min_measurements_correlations})'.format(min_measurements_correlations=self.min_measurements_correlations)
        return string


    @property
    def points_and_results(self):
        values = self._load_data_function()
        sorted_indices = util.math.sort.lex_sorted_indices(values)
        values = values[sorted_indices]
        return values

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    @overrides.overrides
    def points(self):
        return self.points_and_results[:, :-1]

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    @overrides.overrides
    def values(self):
        return self.points_and_results[:, -1]
    



class MeasurementsLadolfi2002(MeasurementsSingleBase):
    
    def __init__(self, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        super().__init__('ladolfi_2002', load_ladolfi_2002, min_measurements_correlations=min_measurements_correlations)


class MeasurementsLadolfi2004(MeasurementsSingleBase):
    
    def __init__(self, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        super().__init__('ladolfi_2004', load_ladolfi_2004, min_measurements_correlations=min_measurements_correlations)


class MeasurementsYoshimura2007(MeasurementsSingleBase):
    
    def __init__(self, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        super().__init__('yoshimura_2007', load_yoshimura_2007, min_measurements_correlations=min_measurements_correlations)


class MeasurementsBase(measurements.universal.data.MeasurementsAnnualPeriodicUnionCache):
    
    def __init__(self, *measurement_list):
        super().__init__(*measurement_list)
        self.standard_deviation_concentration_noise_ratio = measurements.dop.pw.constants.DEVIATION_CONCENTRATION_NOISE_RATIO
    
    
    ## standard_deviation_concentration_noise_ratio 
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.fill_strategy', 'self.min_measurements_standard_deviations'))
    @util.cache.file_based.decorator()
    @overrides.overrides
    def concentation_standard_deviations_for_sample_lsm(self):
        return self.standard_deviation_concentration_noise_ratio * self.average_noise_standard_deviations_for_sample_lsm
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.fill_strategy', 'self.min_measurements_standard_deviations'))
    @util.cache.file_based.decorator()
    @overrides.overrides
    def concentration_standard_deviations(self):
        return self.standard_deviation_concentration_noise_ratio * self.average_noise_standard_deviations
    

    def _fill_strategy_for_kind(self, kind):
        if kind == 'concentration_standard_deviations':
            fill_strategy = self._fill_strategy_for_kind('average_noise_standard_deviations') + '_concentration_noise_ratio_{:g}'.format(self.standard_deviation_concentration_noise_ratio)
        else:
            fill_strategy = super()._fill_strategy_for_kind(kind)
        return fill_strategy


class Measurements(MeasurementsBase):
    
    def __init__(self, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurement_list = [ measurement_class(min_measurements_correlations=min_measurements_correlations) for measurement_class in [MeasurementsLadolfi2002, MeasurementsLadolfi2004, MeasurementsYoshimura2007] ]
        super().__init__(*measurement_list)



class MeasurementsNearWaterLadolfi2002(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):
    
    def __init__(self, water_lsm=None, max_box_distance_to_water=0, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurements = MeasurementsLadolfi2002(min_measurements_correlations=min_measurements_correlations)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)


class MeasurementsNearWaterLadolfi2004(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):
    
    def __init__(self, water_lsm=None, max_box_distance_to_water=0, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurements = MeasurementsLadolfi2004(min_measurements_correlations=min_measurements_correlations)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)


class MeasurementsNearWaterYoshimura2007(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):
    
    def __init__(self, water_lsm=None, max_box_distance_to_water=0, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurements = MeasurementsYoshimura2007(min_measurements_correlations=min_measurements_correlations)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)


class MeasurementsNearWater(MeasurementsBase):
    
    def __init__(self, water_lsm=None, max_box_distance_to_water=0, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurement_list = [ measurement_class(water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations) for measurement_class in [MeasurementsNearWaterLadolfi2002, MeasurementsNearWaterLadolfi2004, MeasurementsNearWaterYoshimura2007] ]
        super().__init__(*measurement_list)

