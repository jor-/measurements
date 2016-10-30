import datetime
import warnings

import numpy as np
import scipy.io

import measurements.po4.wod.constants

import util.io.fs
import util.io.object
import util.datetime

import util.logging
logger = util.logging.logger



class Cruise():

    def __init__(self, file):
        
        ## open netcdf file
        with scipy.io.netcdf.netcdf_file(file, 'r') as f:

            ## read time and data
            day_offset_var = f.variables[measurements.po4.wod.constants.DAY_OFFSET]
            assert day_offset_var.units == measurements.po4.wod.constants.DAY_OFFSET_UNIT
            day_offset = float(day_offset_var.data)
            hours_offset = (day_offset % 1) * 24
            minutes_offset = (hours_offset % 1) * 60
            seconds_offset = (minutes_offset % 1) * 60
    
            day_offset = int(day_offset)
            hours_offset = int(hours_offset)
            minutes_offset = int(minutes_offset)
            seconds_offset = int(seconds_offset)
    
            dt_offset = datetime.timedelta(days=day_offset, hours=hours_offset, minutes=minutes_offset, seconds=seconds_offset)
            dt = measurements.po4.wod.constants.BASE_DATE + dt_offset
            dt_float = util.datetime.datetime_to_float(dt)
    
            ## read coordinates and valid measurements
            try:
                x_var = f.variables[measurements.po4.wod.constants.LON]
                y_var = f.variables[measurements.po4.wod.constants.LAT]
                z_var = f.variables[measurements.po4.wod.constants.DEPTH]
                po4_var = f.variables[measurements.po4.wod.constants.PO4]
                z_flag_var = f.variables[measurements.po4.wod.constants.DEPTH_FLAG]
                po4_flag_var = f.variables[measurements.po4.wod.constants.PO4_FLAG]
                po4_profile_flag_var = f.variables[measurements.po4.wod.constants.PO4_PROFILE_FLAG]
            except KeyError as e:
                missing_key = e.args[0]
                logger.warn('Date with name {} is missing in file {}!'.format(missing_key, file))
                dt_float = float('nan')
                x = float('nan')
                y = float('nan')
                z = np.array([])
                po4 = np.array([])
            else:
                assert x_var.units == measurements.po4.wod.constants.LON_UNIT
                x = float(x_var.data)
                if x == 180:
                    x = -180
                assert x >= -180 and x < 180
                
                assert y_var.units == measurements.po4.wod.constants.LAT_UNIT
                y = float(y_var.data)
                if y == -90 or y == 90:
                    x = 0
                assert y >= -90 and y <= 90
                
                assert z_var.units == measurements.po4.wod.constants.DEPTH_UNIT
                z = z_var.data
                
                assert po4_var.units == measurements.po4.wod.constants.PO4_UNIT
                po4 = po4_var.data
    
                z_flag = z_flag_var.data
                po4_flag = po4_flag_var.data
                po4_profile_flag = po4_profile_flag_var.data
                
    
            ## remove invalid measurements
            if len(po4) > 0:
                valid_mask = np.logical_and(po4_flag == 0, z_flag == 0) * (po4_profile_flag == 0)
                z = z[valid_mask]
                po4 = po4[valid_mask]
    
                valid_mask = po4 != measurements.po4.wod.constants.MISSING_VALUE
                z = z[valid_mask]
                po4 = po4[valid_mask]
    
            ## check values
            if np.any(po4 < 0):
                logger.warn('PO4 in {} is lower then 0!'.format(file))
                valid_mask = po4 > 0
                po4 = po4[valid_mask]
                z = z[valid_mask]
    
            if np.any(z < 0):
                logger.warn('Depth in {} is lower then 0!'.format(file))
                z[z < 0] = 0

        ## save values
        self.dt_float = dt_float
        self.x = x
        self.y = y
        self.z = z
        self.po4 = po4

        logger.debug('Cruise from {} loaded with {:d} values.'.format(file, self.number_of_measurements))


    @property
    def number_of_measurements(self):
        return self.po4.size

    @property
    def year(self):
        year = int(self.dt_float)
        return year

    @property
    def year_fraction(self):
        year_fraction = self.dt_float % 1
        return year_fraction

    def is_year_fraction_in(self, lower_bound=float('-inf'), upper_bound=float('inf')):
        year_fraction = self.year_fraction
        return year_fraction >= lower_bound and year_fraction < upper_bound




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
        logger.debug('Loading all cruises from netcdf files.')

        ## lookup files
        logger.debug('Looking up files in %s.' % data_dir)
        files = util.io.fs.get_files(data_dir, use_absolute_filenames=True)
        logger.debug('%d files found.' % len(files))

        ## load cruises
        logger.debug('Loading cruises from found files.')
        cruises = [Cruise(file) for file in files]
        logger.debug('%d cruises loaded.' % len(cruises))

        ## remove empty cruises
        logger.debug('Removing empty cruises.')
        cruises = [cruise for cruise in cruises if cruise.number_of_measurements > 0]
        logger.debug('%d not empty cruises found.' % len(cruises))

        ## return cruises
        self.cruises = cruises


    def save_cruises_to_pickle_file(self, file):
        logger.debug('Saving cruises at %s.' % file)
        util.io.object.save(file, self.cruises)
        logger.debug('Cruises saved at %s.' % file)


    def load_cruises_from_pickle_file(self, file):
        logger.debug('Loading cruises at %s.' % file)
        self.cruises = util.io.object.load(file)
        logger.debug('Cruises loaded at %s.' % file)
