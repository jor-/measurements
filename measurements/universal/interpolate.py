import abc

import numpy as np

import measurements.util.interpolate

import util.math.interpolate

import util.logging
logger = util.logging.logger




def default_scaling_values(sample_lsm):
    scaling_x = 1
    scaling_y = sample_lsm.x_dim / (sample_lsm.y_dim * 2)
    if scaling_y.is_integer():
        scaling_y = int(scaling_y)
    scaling_t = sample_lsm.x_dim / (sample_lsm.t_dim * 3)
    if scaling_t.is_integer():
        scaling_t = int(scaling_t)
    scaling_z = int(np.floor(sample_lsm.x_dim / sample_lsm.z_dim))
    scaling_values = (scaling_t, scaling_x, scaling_y, scaling_z)
    return scaling_values
    
    

class Interpolator_Annual_Periodic:

    def __init__(self, sample_lsm, scaling_values=None):
        ## sample lsm
        self.sample_lsm = sample_lsm
        
        ## set scaling values
        if scaling_values is None:
            scaling_values = default_scaling_values(sample_lsm)
        self.scaling_values = scaling_values
    
    

    def interpolate_data_for_lsm(self, data, lsm, interpolator_setup):
        logger.debug('Interpolationg data for lsm {} with interpolator_setup {}.'.format(lsm, interpolator_setup))
        
        interpolated_values = measurements.util.interpolate.periodic_with_coordinates(data, lsm.sea_coordinates, self.sample_lsm, scaling_values=self.scaling_values, interpolator_setup=interpolator_setup)

        interpolated_data = np.concatenate((lsm.sea_indices, interpolated_values[:,np.newaxis]), axis=1)
        interpolated_map = lsm.insert_index_values_in_map(interpolated_data, no_data_value=np.inf)
        assert np.all(interpolated_map != np.inf)
        
        util.math.interpolate.change_dim(interpolated_map, 0, lsm.t_dim)
        assert interpolated_map.shape == lsm.dim
        
        return interpolated_map
    


    def interpolate_data_for_points_from_interpolated_lsm_data(self, interpolated_lsm_data, interpolation_points):
        logger.debug('Interpolationg data for points from interpolated data for lsm {}.'.format(self.sample_lsm))
        
        ## get interpolated points and values
        interpolated_lsm_data_mask = ~np.isnan(interpolated_lsm_data)
        interpolated_lsm_values = interpolated_lsm_data[interpolated_lsm_data_mask]
        interpolated_lsm_indices = np.array(np.where(interpolated_lsm_data_mask)).T
        interpolated_lsm_points = self.sample_lsm.map_indices_to_coordinates(interpolated_lsm_indices)
        interpolated_lsm_points_and_values = np.concatenate([interpolated_lsm_points, interpolated_lsm_values[:,np.newaxis]], axis=1)
        
        ## prepare interpolation points: convert to (rounded) int map indices -> convert to coordinates
        interpolation_points_map_indices = self.sample_lsm.coordinates_to_map_indices(interpolation_points, discard_year=True, int_indices=True)
        interpolation_points = self.sample_lsm.map_indices_to_coordinates(interpolation_points_map_indices)

        ## interpolate for points
        interpolated_points_data = measurements.util.interpolate.periodic_with_coordinates(interpolated_lsm_points_and_values, interpolation_points, self.sample_lsm, scaling_values=self.scaling_values, interpolator_setup=(2/min([self.sample_lsm.t_dim,self.sample_lsm.x_dim]),0,0,0))
        
        ## return
        assert np.all(np.isfinite(interpolated_points_data))
        return interpolated_points_data



    def interpolate_data_for_points(self, data, interpolation_points, interpolator_setup):
        logger.debug('Interpolationg data for points with interpolator_setup {}.'.format(interpolator_setup))
        
        ## interpolate for sample lsm
        interpolated_lsm_data = self.interpolate_data_for_lsm(data, self.sample_lsm, interpolator_setup=interpolator_setup)
        
        ## interpolate for points
        interpolated_points_data = self.interpolate_data_for_points_from_interpolated_lsm_data(interpolated_lsm_data, interpolation_points)
        return interpolated_points_data
        
        # ## get interpolated points and values
        # interpolated_lsm_data_mask = ~np.isnan(interpolated_lsm_data)
        # interpolated_lsm_values = interpolated_lsm_data[interpolated_lsm_data_mask]
        # interpolated_lsm_indices = np.array(np.where(interpolated_lsm_data_mask)).T
        # interpolated_lsm_points = self.sample_lsm.map_indices_to_coordinates(interpolated_lsm_indices)
        # interpolated_lsm_points_and_values = np.concatenate([interpolated_lsm_points, interpolated_lsm_values[:,np.newaxis]], axis=1)
        # 
        # ## prepare interpolation points: convert to (rounded) int map indices -> convert to coordinates
        # interpolation_points_map_indices = self.sample_lsm.coordinates_to_map_indices(interpolation_points, discard_year=True, int_indices=True)
        # interpolation_points = self.sample_lsm.map_indices_to_coordinates(interpolation_points_map_indices)

        #   ## interpolate for points
        # interpolated_data = measurements.util.interpolate.periodic_with_coordinates(interpolated_lsm_points_and_values, interpolation_points, self.sample_lsm, scaling_values=self.scaling_values, interpolator_setup=(2/min([self.sample_lsm.t_dim,self.sample_lsm.x_dim]),0,0,0))
        # 
        # ## return
        # assert np.all(np.isfinite(interpolated_data))
        # return interpolated_data
   
        

