import numpy as np

import measurements.land_sea_mask.data
import measurements.po4.wod.data.values
import measurements.po4.wod.constants
import measurements.util.interpolate

import util.cache
import util.math.interpolate


class Interpolator:
    
    def __init__(self, data_function, interpolated_data_dir, interpolated_data_filename):
        self.data_function = data_function
        self.cache = util.cache.HDD_NPY_Cache(interpolated_data_dir)
        self.interpolated_data_filename = interpolated_data_filename
#         self.sample_lsm_t_dim = 48
    
    
    @property
    def sample_lsm(self):
#         return measurements.land_sea_mask.data.LandSeaMaskWOA13R(t_dim=self.sample_lsm_t_dim)
        return measurements.po4.wod.constants.SAMPLE_LSM
    
    @property
    def data(self):
        m = measurements.po4.wod.data.values.measurement_dict()
        m.categorize_indices_to_lsm(self.sample_lsm, discard_year=True)
        return self.data_function(m)
    
    

    def sample_data_for_lsm(self, lsm, no_data_value=np.inf):
        data = np.copy(self.data)
        data[:,:-1] = lsm.coordinates_to_map_indices(data[:,:-1])
        for i in range(len(data)):
            data[i, 3] = min(data[i, 3], lsm.z_dim - 1)
        data_map = lsm.insert_index_values_in_map(data, no_data_value=no_data_value)
        return data_map



    def interpolated_data_for_lsm_calculate(self, lsm, interpolator_setup=(0.1, 1, 0.0, 0)):
        data = np.copy(self.data)
        
        interpolated_values = measurements.util.interpolate.periodic_with_coordinates(data, lsm.sea_coordinates, self.sample_lsm, interpolator_setup=interpolator_setup)
        
        interpolated_data = np.concatenate((lsm.sea_indices, interpolated_values[:,np.newaxis]), axis=1)
        interpolated_map = lsm.insert_index_values_in_map(interpolated_data, no_data_value=np.inf)
        
        assert np.all(interpolated_map != np.inf)
        return interpolated_map


    def interpolated_data_for_lsm(self, lsm, interpolator_setup=(0.1, 1, 0.0, 0)):
        filename = self.interpolated_data_filename.format(lsm, str(interpolator_setup).replace(' ','').replace('(','').replace(')',''))
        function = lambda :self.interpolated_data_for_lsm_calculate(lsm, interpolator_setup=interpolator_setup)
        return self.cache.get_value(filename, function)



    def interpolated_data_for_points_calculate(self, interpolator_setup=(0.1, 1, 0.0, 0)):
        data = np.copy(self.data)
        
        interpolation_points = measurements.po4.wod.data.values.measurement_points()
        interpolated_data = measurements.util.interpolate.periodic_with_coordinates(data, interpolation_points, self.sample_lsm, interpolator_setup=interpolator_setup)
        
        return interpolated_data


    def interpolated_data_for_points(self, interpolator_setup=(0.1, 1, 0.0, 0)):
        filename = self.interpolated_data_filename.format('lexsorted_points', str(interpolator_setup).replace(' ','').replace('(','').replace(')',''))
        function = lambda :self.interpolated_data_for_points_calculate(interpolator_setup=interpolator_setup)
        return self.cache.get_value(filename, function)


    def data_for_points(self, interpolator_setup):
        return self.interpolated_data_for_points(interpolator_setup=interpolator_setup)
    
    def data_for_TMM(self, t_dim, interpolator_setup):
        lsm = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=self.sample_lsm.t_dim)
        data = self.interpolated_data_for_lsm(lsm, interpolator_setup=interpolator_setup)
        return util.math.interpolate.change_dim(data, 0, t_dim)
    
    def data_for_WOA13(self, t_dim, interpolator_setup):
        lsm = measurements.land_sea_mask.data.LandSeaMaskWOA13(t_dim=self.sample_lsm.t_dim)
        data = self.interpolated_data_for_lsm(lsm, interpolator_setup=interpolator_setup)
        return util.math.interpolate.change_dim(data, 0, t_dim)
    
    def data_for_WOA13R(self, t_dim, interpolator_setup):
        lsm = measurements.land_sea_mask.data.LandSeaMaskWOA13R(t_dim=self.sample_lsm.t_dim)
        data = self.interpolated_data_for_lsm(lsm, interpolator_setup=interpolator_setup)
        return util.math.interpolate.change_dim(data, 0, t_dim)
    
