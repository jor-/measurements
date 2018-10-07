import calendar
import datetime
import os.path

import numpy as np
import overrides

import util.math.sort
import util.cache.file
import util.cache.memory
import util.logging

import measurements.universal.data
import measurements.dop.constants


# data load functions

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
    data = data[data[:, 4] == valid_data_flag]
    data = data[:, :4]
    data = data[data[:, 3] >= 0]
    data = data[np.logical_not(np.isnan(data[:, 3]))]

    # move columns in order long, lat, depth, dop
    data[:, [0, 1]] = data[:, [1, 0]]

    # sort data in priority by long, lat, depth
    data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]

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
    t = np.arange(n) / (n - 1) * (end_year_float - start_year_float) + start_year_float
    t = t.reshape([n, 1])
    data = np.concatenate((t, data), axis=1)

    util.logging.debug('{} DOP data sets loaded from {}.'.format(data.shape[0], measurement_file))

    return data


@util.cache.file.decorator(cache_file_function=lambda: os.path.join(measurements.dop.constants.LADOLFI_2002_DIR, measurements.dop.constants.DATA_FILENAME))
def load_ladolfi_2002():
    from measurements.dop.constants import LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_START_DATE, LADOLFI_2002_END_DATE, LADOLFI_2002_VALID_DATA_FLAG
    return prepare_ladolfi_data(LADOLFI_2002_MEASUREMENT_FILE, LADOLFI_2002_START_DATE, LADOLFI_2002_END_DATE, LADOLFI_2002_VALID_DATA_FLAG)


@util.cache.file.decorator(cache_file_function=lambda: os.path.join(measurements.dop.constants.LADOLFI_2004_DIR, measurements.dop.constants.DATA_FILENAME))
def load_ladolfi_2004():
    from measurements.dop.constants import LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_START_DATE, LADOLFI_2004_END_DATE, LADOLFI_2004_VALID_DATA_FLAG
    return prepare_ladolfi_data(LADOLFI_2004_MEASUREMENT_FILE, LADOLFI_2004_START_DATE, LADOLFI_2004_END_DATE, LADOLFI_2004_VALID_DATA_FLAG)


def prepare_yoshimura_data(measurement_file):
    # columns: lat, long, date, time, dop

    # calculate time
    def convert_date_to_year(value_bytes):
        value_string = value_bytes.decode()
        d = datetime.datetime.strptime(value_string, '%d-%B-%Y').date()
        year = d.year
        util.logging.debug('Convering: "{}" is in the year {}.'.format(value_bytes, year))
        return year

    def convert_date_to_number_of_days_in_year(value_bytes):
        value_string = value_bytes.decode()
        d = datetime.datetime.strptime(value_string, '%d-%B-%Y').date()
        number_of_days = d.timetuple().tm_yday
        util.logging.debug('Convering: "{}" is the {}. day in a year.'.format(value_bytes, number_of_days))
        return number_of_days

    def convert_date_to_number_of_all_days_in_year(value_bytes):
        value_string = value_bytes.decode()
        d = datetime.datetime.strptime(value_string, '%d-%B-%Y').date()
        number_of_days_in_year = calendar.isleap(d.year) + 365
        util.logging.debug('Convering: The year of "{}" has {} days.'.format(value_bytes, number_of_days_in_year))
        return number_of_days_in_year

    def convert_time_to_day_frac(value_bytes):
        value_string = value_bytes.decode()
        dt = datetime.datetime.strptime(value_string, '%H:%M')
        hour_frac = dt.hour + dt.minute / 60
        day_frac = hour_frac / 24
        util.logging.debug('Convering: The time "{}" is {} of a day.'.format(value_bytes, day_frac))
        return day_frac

    year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_year})

    number_of_days_in_year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_number_of_days_in_year})
    number_of_all_days_in_year = np.loadtxt(measurement_file, usecols=(2,), converters={2: convert_date_to_number_of_all_days_in_year})
    day_frac = np.loadtxt(measurement_file, usecols=(3,), converters={3: convert_time_to_day_frac})
    year_frac = (number_of_days_in_year + day_frac) / number_of_all_days_in_year

    time = year + year_frac

    # load lat, long and dop
    lat = np.loadtxt(measurement_file, usecols=(0,))
    long = np.loadtxt(measurement_file, usecols=(1,))
    dop = np.loadtxt(measurement_file, usecols=(4,))
    depth = dop * 0

    # concatenate columns in order long, lat, depth, dop
    data = np.concatenate((time[:, np.newaxis], long[:, np.newaxis], lat[:, np.newaxis], depth[:, np.newaxis], dop[:, np.newaxis]), axis=1)

    util.logging.debug('{} DOP data sets loaded from {}.'.format(data.shape[0], measurement_file))

    return data


@util.cache.file.decorator(cache_file_function=lambda: os.path.join(measurements.dop.constants.YOSHIMURA_2007_DIR, measurements.dop.constants.DATA_FILENAME))
def load_yoshimura_2007():
    from measurements.dop.constants import YOSHIMURA_2007_MEASUREMENT_FILE
    return prepare_yoshimura_data(YOSHIMURA_2007_MEASUREMENT_FILE)


# measurement classes

class MeasurementsSingleBase(measurements.universal.data.MeasurementsAnnualPeriodicCache):

    def __init__(self, data_set_name, load_data_function,
                 sample_lsm=measurements.dop.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.dop.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.dop.constants.CORRELATION_MIN_MEASUREMENTS):

        self._load_data_function = load_data_function

        tracer = 'dop'
        data_set_name = data_set_name

        if sample_lsm is None:
            sample_lsm = measurements.dop.constants.SAMPLE_LSM
        if min_standard_deviation is None:
            min_standard_deviation = measurements.dop.constants.STANDARD_DEVIATION_MIN_VALUE
        if min_measurements_correlation is None:
            min_measurements_correlation = measurements.dop.constants.CORRELATION_MIN_MEASUREMENTS

        super().__init__(
            tracer=tracer,
            data_set_name=data_set_name,
            sample_lsm=sample_lsm,
            min_standard_deviation=min_standard_deviation,
            min_measurements_correlation=min_measurements_correlation)

    @property
    def points_and_results(self):
        values = self._load_data_function()
        sorted_indices = util.math.sort.lex_sorted_indices(values)
        values = values[sorted_indices]
        return values

    @property
    @overrides.overrides
    def points(self):
        return self.points_and_results[:, :-1]

    @property
    @overrides.overrides
    def values(self):
        return self.points_and_results[:, -1]


class MeasurementsLadolfi2002(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):

    def __init__(self,
                 sample_lsm=measurements.dop.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.dop.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.dop.constants.CORRELATION_MIN_MEASUREMENTS,
                 water_lsm=None,
                 max_box_distance_to_water=None):

        measurements_base = MeasurementsSingleBase(
            'ladolfi_2002', load_ladolfi_2002,
            sample_lsm=sample_lsm,
            min_standard_deviation=min_standard_deviation,
            min_measurements_correlation=min_measurements_correlation)

        super().__init__(measurements_base,
                         water_lsm=water_lsm,
                         max_box_distance_to_water=max_box_distance_to_water)


class MeasurementsLadolfi2004(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):

    def __init__(self,
                 sample_lsm=measurements.dop.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.dop.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.dop.constants.CORRELATION_MIN_MEASUREMENTS,
                 water_lsm=None,
                 max_box_distance_to_water=None):

        measurements_base = MeasurementsSingleBase(
            'ladolfi_2004', load_ladolfi_2004,
            sample_lsm=sample_lsm,
            min_standard_deviation=min_standard_deviation,
            min_measurements_correlation=min_measurements_correlation)

        super().__init__(measurements_base,
                         water_lsm=water_lsm,
                         max_box_distance_to_water=max_box_distance_to_water)


class MeasurementsYoshimura2007(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):

    def __init__(self,
                 sample_lsm=measurements.dop.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.dop.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.dop.constants.CORRELATION_MIN_MEASUREMENTS,
                 water_lsm=None,
                 max_box_distance_to_water=None):

        measurements_base = MeasurementsSingleBase(
            'yoshimura_2007', load_yoshimura_2007,
            sample_lsm=sample_lsm,
            min_standard_deviation=min_standard_deviation,
            min_measurements_correlation=min_measurements_correlation)

        super().__init__(measurements_base,
                         water_lsm=water_lsm,
                         max_box_distance_to_water=max_box_distance_to_water)


class MeasurementsBase(measurements.universal.data.MeasurementsAnnualPeriodicUnionCache):

    def __init__(self,
                 sample_lsm=measurements.dop.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.dop.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.dop.constants.CORRELATION_MIN_MEASUREMENTS):
        measurement_list = [measurement_class(sample_lsm=sample_lsm, min_standard_deviation=min_standard_deviation, min_measurements_correlation=min_measurements_correlation)
                            for measurement_class in [MeasurementsLadolfi2002, MeasurementsLadolfi2004, MeasurementsYoshimura2007]]
        super().__init__(*measurement_list)
        self.standard_deviation_concentration_noise_ratio = measurements.dop.constants.STANDARD_DEVIATION_CONCENTRATION_NOISE_RATIO
        self.fill_strategy = 'point_average'

    # standard_deviation_concentration_noise_ratio

    @util.cache.memory.method_decorator(dependency=('self.fill_strategy', 'self.min_measurements_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def concentration_standard_deviations_for_sample_lsm(self):
        return self.standard_deviation_concentration_noise_ratio * self.average_noise_standard_deviations_for_sample_lsm()

    @property
    @util.cache.memory.method_decorator(dependency=('self.fill_strategy', 'self.min_measurements_standard_deviation'))
    @util.cache.file.decorator()
    @overrides.overrides
    def concentration_standard_deviations(self):
        return self.standard_deviation_concentration_noise_ratio * self.average_noise_standard_deviations

    def _fill_strategy_for_kind(self, kind):
        if kind == 'concentration_standard_deviations':
            fill_strategy = self._fill_strategy_for_kind('average_noise_standard_deviations') + '_ratio_{:g}'.format(self.standard_deviation_concentration_noise_ratio)
        else:
            fill_strategy = super()._fill_strategy_for_kind(kind)
        return fill_strategy


class Measurements(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):

    def __init__(self,
                 sample_lsm=measurements.dop.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.dop.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.dop.constants.CORRELATION_MIN_MEASUREMENTS,
                 water_lsm=None,
                 max_box_distance_to_water=None):
        measurements = MeasurementsBase(sample_lsm=sample_lsm, min_standard_deviation=min_standard_deviation, min_measurements_correlation=min_measurements_correlation)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)
