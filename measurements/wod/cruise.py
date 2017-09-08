import abc
import datetime

import scipy.io

import util.io.fs
import util.io.object
import util.logging

import measurements.wod.constants


class UnitError(ValueError):
    def __init__(self, name, unit, wanted_unit):
        message = 'The variable "{}" has unit "{}" but unit "{}" was needed.'.format(name, unit, wanted_unit)
        self.message = message
        self.unit = unit
        self.wanted_unit = wanted_unit
        super().__init__(message)


class InvalidValueError(ValueError):
    def __init__(self, name, value):
        message = 'The value "{}" of variable "{}" is invalid.'.format(value, name)
        self.message = message
        self.value = value
        super().__init__(message)


class MissingValueError(KeyError):
    def __init__(self, name):
        message = 'The variable "{}" is missing.'.format(name)
        self.message = message
        super().__init__(message)


class Cruise(metaclass=abc.ABCMeta):

    DATE_VARIABLE = measurements.wod.constants.DATE_VARIABLE
    TIME_VARIABLE = measurements.wod.constants.TIME_VARIABLE
    START_TIME_VARIABLE = measurements.wod.constants.START_TIME_VARIABLE
    DATE_TIME_VARIABLE = measurements.wod.constants.DATE_TIME_VARIABLE
    LATITUDE_VARIABLE = measurements.wod.constants.LATITUDE_VARIABLE
    LONGITUDE_VARIABLE = measurements.wod.constants.LONGITUDE_VARIABLE

    def __init__(self, file):
        util.logging.debug('Loading cruise from {}.'.format(file))
        with scipy.io.netcdf.netcdf_file(file, 'r') as file:
            self.time = self._load_time(file)
            longitude, latitude = self._load_longitude_and_latitude(file)
            self.longitude = longitude
            self.latitude = latitude
            depths, values = self._load_depths_and_values(file)
            self.depths = depths
            self.values = values
        util.logging.debug('Cruise from {} loaded with {:d} values.'.format(file, self.number_of_measurements))

    @property
    def number_of_measurements(self):
        return len(self.values)

    # *** load functions *** #

    def _load_value_from_file(self, file, name, unit=None, unit_name='units'):
        try:
            var = file.variables[name]
        except KeyError as e:
            raise MissingValueError(name) from e
        else:
            if unit is not None:
                var_unit = getattr(var, unit_name)
                if var_unit != unit:
                    raise UnitError(name, var_unit, unit)
            return var.data

    def _load_value_from_file_with_variable_information_dict(self, file, variable_information_dict):
        return self._load_value_from_file(file, variable_information_dict['name'], unit=variable_information_dict['unit'], unit_name=variable_information_dict['unit_name'])

    def _load_time(self, file):
        # load functions
        def load_date(file):
            date_int = self._load_value_from_file_with_variable_information_dict(file, self.DATE_VARIABLE)
            try:
                date_str = str(date_int)
                assert len(date_str) == 8
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                date = datetime.datetime(year, month, day)
            except ValueError as e:
                raise InvalidValueError(self.DATE_VARIABLE['name'], date_int) from e
            return date

        def _load_hours(file, variable_information_dict):
            time = self._load_value_from_file_with_variable_information_dict(file, variable_information_dict)
            try:
                hours = float(time)
                time = datetime.timedelta(hours=hours)
            except ValueError as e:
                raise InvalidValueError(variable_information_dict['name'], time) from e
            if not 0 <= hours < 24:
                raise InvalidValueError(variable_information_dict['name'], time)
            return time

        def load_time(file):
            return _load_hours(file, self.TIME_VARIABLE)

        def load_start_time(file):
            return _load_hours(file, self.START_TIME_VARIABLE)

        def load_date_time(file):
            date_time = self._load_value_from_file_with_variable_information_dict(file, self.DATE_TIME_VARIABLE)
            try:
                days = float(date_time)
                date_time_offset = datetime.timedelta(days=days)
                date_time = measurements.wod.constants.DATE_TIME_BASE + date_time_offset
            except ValueError as e:
                raise InvalidValueError(self.DATE_TIME_VARIABLE['name'], date_time) from e
            return date_time

        # calculate date and time
        try:
            date = load_date(file)
        except (InvalidValueError, MissingValueError) as e:
            util.logging.warn(e.message)
            date = None
        else:
            try:
                time = load_time(file)
            except (InvalidValueError, MissingValueError) as e:
                util.logging.warn(e.message)
                try:
                    time = load_start_time(file)
                except (InvalidValueError, MissingValueError) as e:
                    util.logging.warn(e.message)
                    time = None

        if date is None or time is None:
            date_time = load_date_time(file)
        else:
            date_time = date + time

        # convert to float value
        year = date_time.year
        year_start = datetime.datetime(year, 1, 1)
        next_year_start = datetime.datetime(year + 1, 1, 1)
        float_fraction = (date_time - year_start).total_seconds() / ((next_year_start - year_start).total_seconds())
        date_time_float = year + float_fraction
        return date_time_float

    def _load_longitude_and_latitude(self, file):
        lon = self._load_value_from_file_with_variable_information_dict(file, self.LONGITUDE_VARIABLE)
        lon = float(lon)
        if lon == 180:
            lon = -180
        if not (lon >= -180 and lon < 180):
            raise InvalidValueError(self.LONGITUDE_VARIABLE['name'], lon)

        lat = self._load_value_from_file_with_variable_information_dict(file, self.LATITUDE_VARIABLE)
        lat = float(lat)
        if lat == -90 or lat == 90:
            lon = 0
        if not (lat >= -90 and lat <= 90):
            raise InvalidValueError(self.LATITUDE_VARIABLE['name'], lat)

        return lon, lat

    @abc.abstractmethod
    def _load_depths_and_values(self, file):
        raise NotImplementedError()


class CruisesCollection():

    def __init__(self, cruises_class):
        self.cruises_class = cruises_class
        self._cruises = []

    @property
    def cruises(self):
        return self._cruises

    @cruises.setter
    def cruises(self, cruises):
        self._cruises = list(cruises)

    # *** add cruises *** #

    def add_cruise(self, cruise, only_if_nonempty=True):
        if not only_if_nonempty or cruise.number_of_measurements > 0:
            if isinstance(cruise, self.cruises_class):
                self._cruises.append(cruise)
            else:
                raise ValueError('Cruise {} is not of class {}.'.format(cruise, self.cruises_class))

    def add_cruise_from_netcdf_file(self, file):
        cruise = self.cruises_class(file)
        self.add_cruise(cruise)

    def add_cruises_from_netcdf_files_in_directory(self, directory):
        util.logging.debug('Loading all cruises from netcdf files in {}.'.format(directory))
        files = util.io.fs.get_files(directory, use_absolute_filenames=True)
        util.logging.debug('{} files found.'.format(len(files)))
        for file in files:
            self.add_cruise_from_netcdf_file(file)

    def add_cruises_from_netcdf_files_in_tar_file(self, file):
        import tarfile
        util.logging.debug('Loading all cruises from tar file {}.'.format(file))
        with tarfile.open(file, mode='r') as tar_file_object:
            for member in tar_file_object:
                util.logging.debug('Loading cruise from netcdf file {}.'.format(member.name))
                with tar_file_object.extractfile(member) as netcdf_file:
                    self.add_cruise_from_netcdf_file(netcdf_file)

    # *** save and load *** #

    def save(self, file):
        util.logging.debug('Saving cruises to {}.'.format(file))
        util.io.object.save(file, self)
        util.logging.debug('Cruises saved to {}.'.format(file))

    @staticmethod
    def load(file):
        util.logging.debug('Loading cruises from {}.'.format(file))
        loaded_collection = util.io.object.load(file)
        util.logging.debug('Cruises loaded from {}.'.format(file))
        return loaded_collection

    # *** container type methods *** #

    def __len__(self):
        return self.cruises.__len__()

    def __length_hint__(self):
        return self.cruises.__length_hint__()

    def __getitem__(self, key):
        return self.cruises.__getitem__(key)

    def __missing__(self, key):
        return self.cruises.__missing__(key)

    def __setitem__(self, key, value):
        return self.cruises.__setitem__(key, value)

    def __delitem__(self, key):
        return self.cruises.__delitem__(key)

    def __reversed__(self):
        return self.cruises.__reversed__()

    def __contains__(self, item):
        return self.cruises.__contains__(item)
