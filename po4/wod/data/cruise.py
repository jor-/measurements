import scipy.io
import numpy as np

import datetime
import warnings

import ndop.model.data

import util.io
import util.datetime

from .constants import CRUISES_FILE, DATA_DIR

import logging
logger = logging.getLogger(__name__)


class Cruise():
    
    def __init__(self, file):
        from . import constants
        
        logger.debug('Loading cruise from {}.'.format(file))
        
        ## open netcdf file
        f = scipy.io.netcdf.netcdf_file(file, 'r')
        
        ## read time and data
        day_offset = float(f.variables[constants.DAY_OFFSET].data)
        hours_offset = (day_offset % 1) * 24
        minutes_offset = (hours_offset % 1) * 60
        seconds_offset = (minutes_offset % 1) * 60
        
        day_offset = int(day_offset)
        hours_offset = int(hours_offset)
        minutes_offset = int(minutes_offset)
        seconds_offset = int(seconds_offset)
        
        dt_offset = datetime.timedelta(days=day_offset, hours=hours_offset, minutes=minutes_offset, seconds=seconds_offset)
        dt = constants.BASE_DATE + dt_offset
        dt_float = util.datetime.datetime_to_float(dt)
        
        self.dt_float = dt_float
        
        ## read coordinates and valid measurements
        try:
            self.x = float(f.variables[constants.LON].data)
            self.y = float(f.variables[constants.LAT].data)
            
            z = f.variables[constants.DEPTH].data
            po4 = f.variables[constants.PO4].data
            
            z_flag = f.variables[constants.DEPTH_FLAG].data
            po4_flag = f.variables[constants.PO4_FLAG].data
            po4_profile_flag = f.variables[constants.PO4_PROFILE_FLAG].data
        except KeyError as e:
            missing_key = e.args[0]
            warnings.warn('Date with name {} is missing in file {}!'.format(missing_key, file))
            z = np.array([])
            po4 = np.array([])
        
        ## remove invalid measurements
        if len(po4) > 0:
            valid_mask = np.logical_and(po4_flag == 0, z_flag == 0) * (po4_profile_flag == 0)
            z = z[valid_mask]
            po4 = po4[valid_mask]
            
            valid_mask = po4 != constants.MISSING_VALUE
            z = z[valid_mask]
            po4 = po4[valid_mask]
        
        ## check values
        if np.any(po4 < 0):
            warnings.warn('PO4 in {} is lower then 0!'.format(file))
            valid_mask = po4 > 0
            po4 = po4[valid_mask]
            z = z[valid_mask]
        
        if np.any(z < 0):
            warnings.warn('Depth in {} is lower then 0!'.format(file))
            z[z < 0] = 0
        
        ## save values
        self.z = z
        self.po4 = po4
        
        ## close file
        f.close()
        
        logger.debug('Cruise from {} loaded.'.format(file))
    
    @property
    def number_of_measurements(self):
        return self.po4.size
    
    @property
    def land_sea_mask(self):
        try:
            return self.__land_sea_mask
        except AttributeError:
            raise Exception('Land sea mask is not set.')
    
    @land_sea_mask.setter
    def land_sea_mask(self, land_sea_mask):
        self.__land_sea_mask = land_sea_mask
        self.__spatial_indices = None
    
    @property
    def spatial_indices(self):
        try:
            indices = self.__spatial_indices
        except AttributeError:
            indices = None
        
        if indices == None:
            logger.debug('Calculating spatil indices')
            
            land_sea_mask = self.land_sea_mask
            x = self.x
            y = self.y
            z = self.z
            
            m = z.size
            
            indices = np.empty((m, 3), dtype=np.uint16)
            
            for i in range(m):
                indices[i] = ndop.model.data.get_spatial_index(x, y, z[i], land_sea_mask)
            
            self.__spatial_indices = indices
        
        return indices
    
    @spatial_indices.setter
    def spatial_indices(self, spatial_indices):
        self.__spatial_indices = spatial_indices
    
    
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





class Cruise_Collection():
    
    def __init__(self, cruises=None):
        self.__cruises = cruises
    
    
    @property
    def cruises(self):
        try:
            cruises = self.__cruises
        except AttributeError:
            cruises = None
        
        if cruises == None:
            try:
                self.load_cruises_from_pickle_file()
            except (OSError, IOError):
                self.load_cruises_from_netcdf()
                self.save_cruises_to_pickle_file()
            
            cruises = self.cruises
        
        return cruises
    
    @cruises.setter
    def cruises(self, cruises):
        self.__cruises = cruises
    
    
    def calculate_spatial_indices(self):
        cruises = self.cruises
        
        logger.debug('Calculating spatial indices for %d cruises.' % len(cruises))
        
        land_sea_mask = ndop.model.data.load_land_sea_mask()
        
        for cruise in cruises:
            cruise.land_sea_mask = land_sea_mask
            cruise.spatial_indices
        
        logger.debug('For %d cruises spatial indices calculted.' % len(cruises))
    
    
    def load_cruises_from_netcdf(self, data_dir=DATA_DIR):
        logger.debug('Loading all cruises from netcdf files.')
        
        ## lookup files
        logger.debug('Looking up files in %s.' % data_dir)
        files = util.io.get_files(data_dir)
        logger.debug('%d files found.' % len(files))
        
        ## load land sea mask
        land_sea_mask = ndop.model.data.load_land_sea_mask()
        
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
    
    
    
    def save_cruises_to_pickle_file(self, file=CRUISES_FILE):
        logger.debug('Saving cruises at %s.' % file)
        util.io.save_object(self.cruises, file)
        logger.debug('Cruises saved at %s.' % file)
    
    
    def load_cruises_from_pickle_file(self, file=CRUISES_FILE):
        logger.debug('Loading cruises at %s.' % file)
        self.cruises = util.io.load_object(file)
        logger.debug('Cruises loaded at %s.' % file)
