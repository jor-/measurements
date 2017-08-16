import datetime

import numpy as np
import scipy.io

import measurements.po4.wod.constants

import util.io.fs
import util.io.object
import util.datetime

import util.logging


class UnitError(ValueError):
    def __init__(self, name, unit, wanted_unit):
        message = 'The variable {} has unit {} but unit {} was needed!'.format(name, unit, wanted_unit)
        super.__init__(message)

class MissingError(KeyError):
    def __init__(self, name):
        message = 'The variable {} is missing!'.format(name)
        super.__init__(message)



class Cruise():

    def __init__(self, file):

        def value_in_file(file, name, unit=None):
            try:
                var = file.variables[name]
            except KeyError as e:
                raise MissingError(name) from e
            else:
                if unit is not None and var.unit != unit:
                    raise UnitError(name, var.unit, unit)
                return var.data

        ## open netcdf file
        with scipy.io.netcdf.netcdf_file(file, 'r') as file:

            ## read time and data
            try:
                day_offset = value_in_file(file, measurements.po4.wod.constants.DAY_OFFSET, unit=measurements.po4.wod.constants.DAY_OFFSET_UNIT)
            except MissingError:
                time = float('nan')
            else:
                day_offset = float(day_offset)
                hours_offset = (day_offset % 1) * 24
                minutes_offset = (hours_offset % 1) * 60
                seconds_offset = (minutes_offset % 1) * 60

                day_offset = int(day_offset)
                hours_offset = int(hours_offset)
                minutes_offset = int(minutes_offset)
                seconds_offset = int(seconds_offset)

                dt_offset = datetime.timedelta(days=day_offset, hours=hours_offset, minutes=minutes_offset, seconds=seconds_offset)
                dt = measurements.po4.wod.constants.BASE_DATE + dt_offset
                time = util.datetime.datetime_to_float(dt)

            ## read coordinates and depth
            try:
                lon = value_in_file(file, measurements.po4.wod.constants.LON, unit=measurements.po4.wod.constants.LON_UNIT)
            except MissingError:
                lon = float('nan')
            else:
                lon = float(lon)
                if lon == 180:
                    lon = -180
                assert lon >= -180 and lon < 180

            try:
                lat = value_in_file(file, measurements.po4.wod.constants.LAT, unit=measurements.po4.wod.constants.LAT_UNIT)
            except MissingError:
                lat = float('nan')
            else:
                lat = float(lat)
                if lat == -90 or lat == 90:
                    lon = 0
                assert lat >= -90 and lat <= 90

            try:
                depth = value_in_file(file, measurements.po4.wod.constants.DEPTH, unit=measurements.po4.wod.constants.DEPTH_UNIT)
            except MissingError:
                depth = np.array([])

            ## read value
            try:
                values = value_in_file(file, measurements.po4.wod.constants.PO4, unit=measurements.po4.wod.constants.PO4_UNIT)
            except MissingError:
                values = np.array([])

            ## remove invalid measurements
            if len(values) > 0:
                z_flag = value_in_file(file, measurements.po4.wod.constants.DEPTH_FLAG)
                po4_flag = value_in_file(file, measurements.po4.wod.constants.PO4_FLAG)
                po4_profile_flag = value_in_file(file, measurements.po4.wod.constants.PO4_PROFILE_FLAG)

                valid_mask = np.logical_and(po4_flag == 0, z_flag == 0) * (po4_profile_flag == 0)
                valid_mask = np.logical_and(valid_mask, values != measurements.po4.wod.constants.MISSING_VALUE)
                depth = depth[valid_mask]
                values = values[valid_mask]


            ## check values
            if np.any(values < 0):
                util.logging.warn('Values in {} are lower then 0!'.format(file))
                valid_mask = values > 0
                values = values[valid_mask]
                depth = depth[valid_mask]

            if np.any(depth < 0):
                util.logging.warn('Depth in {} is lower then 0!'.format(file))
                depth[depth < 0] = 0


        ## save values
        self.time = time
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.values = values

        util.logging.debug('Cruise from {} loaded with {:d} values.'.format(file, self.number_of_measurements))



class CruiseCollection():

    def __init__(self, cruises=None):
        self.__cruises = cruises


    @property
    def cruises(self):
        try:
            cruises = self.__cruises
        except AttributeError:
            cruises = None

        return cruises

    @cruises.setter
    def cruises(self, cruises):
        self.__cruises = cruises


    def load_cruises_from_netcdf_files(self, data_dir):
        util.logging.debug('Loading all cruises from netcdf files.')

        ## lookup files
        util.logging.debug('Looking up files in %s.' % data_dir)
        files = util.io.fs.get_files(data_dir, use_absolute_filenames=True)
        util.logging.debug('%d files found.' % len(files))

        ## load cruises
        util.logging.debug('Loading cruises from found files.')
        cruises = [Cruise(file) for file in files]
        util.logging.debug('%d cruises loaded.' % len(cruises))

        ## remove empty cruises
        util.logging.debug('Removing empty cruises.')
        cruises = [cruise for cruise in cruises if cruise.number_of_measurements > 0]
        util.logging.debug('%d not empty cruises found.' % len(cruises))

        ## return cruises
        self.cruises = cruises


    def save_cruises_to_pickle_file(self, file):
        util.logging.debug('Saving cruises at %s.' % file)
        util.io.object.save(file, self.cruises)
        util.logging.debug('Cruises saved at %s.' % file)


    def load_cruises_from_pickle_file(self, file):
        util.logging.debug('Loading cruises at %s.' % file)
        self.cruises = util.io.object.load(file)
        util.logging.debug('Cruises loaded at %s.' % file)
