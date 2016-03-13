import numpy as np

import measurements.land_sea_mask.data
import measurements.po4.wod.data.values
import measurements.po4.wod.constants
import measurements.util.interpolate

import util.cache
import util.math.interpolate


class Interpolator:

    def __init__(self, interpolated_data_dir, interpolated_data_filename, sample_lsm=measurements.po4.wod.constants.SAMPLE_LSM, scaling_values=None, default_interpolator_setups=None):
        self.cache = util.cache.HDD_NPY_Cache(interpolated_data_dir)
        self.interpolated_data_filename = interpolated_data_filename
        self.sample_lsm = sample_lsm
        if scaling_values is None:
            scaling_x = 1
            scaling_y = sample_lsm.x_dim / (sample_lsm.y_dim * 2)
            if scaling_y.is_integer():
                scaling_y = int(scaling_y)
            scaling_t = sample_lsm.x_dim / (sample_lsm.t_dim * 3)
            if scaling_t.is_integer():
                scaling_t = int(scaling_t)
            scaling_z = int(np.floor(sample_lsm.x_dim / sample_lsm.z_dim))
            scaling_values = (scaling_t, scaling_x, scaling_y, scaling_z)
        self.scaling_values = scaling_values
        if default_interpolator_setups is None:
            default_interpolator_setups = {}
        self.default_interpolator_setups = default_interpolator_setups
    
    
    def get_interpolator_setup(self, interpolator_setup=None):
        if interpolator_setup is None:
            interpolator_setup = self.default_interpolator_setups[str(self.sample_lsm)]
        return interpolator_setup


    @staticmethod
    def _tuple_to_str(t):
        if t is None:
            return str(t)
        else:
            return ','.join(map(str, t))
        

    @property
    def data(self):
        raise NotImplementedError()


    def sample_data_for_lsm(self, lsm, no_data_value=np.inf):
        data = np.copy(self.data)
        data[:, :-1] = lsm.coordinates_to_map_indices(data[:, :-1])
        for i in range(len(data)):
            data[i, 3] = min(data[i, 3], lsm.z_dim - 1)
        data_map = lsm.insert_index_values_in_map(data, no_data_value=no_data_value)
        return data_map



    def interpolated_data_for_lsm_calculate(self, lsm, interpolator_setup):
        data = np.copy(self.data)

        interpolated_values = measurements.util.interpolate.periodic_with_coordinates(data, lsm.sea_coordinates, self.sample_lsm, scaling_values=self.scaling_values, interpolator_setup=interpolator_setup)

        interpolated_data = np.concatenate((lsm.sea_indices, interpolated_values[:,np.newaxis]), axis=1)
        interpolated_map = lsm.insert_index_values_in_map(interpolated_data, no_data_value=np.inf)
        assert np.all(interpolated_map != np.inf)
        
        util.math.interpolate.change_dim(interpolated_map, 0, lsm.t_dim)
        assert interpolated_map.shape == lsm.dim
        
        return interpolated_map


    def interpolated_data_for_lsm(self, lsm, interpolator_setup=None):
        interpolator_setup = self.get_interpolator_setup(interpolator_setup)
        filename = self.interpolated_data_filename.format(points=lsm, interpolator_setup=self._tuple_to_str(interpolator_setup), scaling_values=self._tuple_to_str(self.scaling_values))
        function = lambda :self.interpolated_data_for_lsm_calculate(lsm, interpolator_setup=interpolator_setup)
        return self.cache.get_value(filename, function)


    def interpolated_data_for_points_calculate(self, interpolator_setup):
        ## prepare interpolation points (convert to (rounded) int map indices -> convert to coordinates)
        interpolation_points = measurements.po4.wod.data.values.points()
        interpolation_points_map_indices = self.sample_lsm.coordinates_to_map_indices(interpolation_points, discard_year=True, float_indices=False)
        interpolation_points = self.sample_lsm.map_indices_to_coordinates(interpolation_points_map_indices)

        ## prepare interpolated points and values (map indices of values -> convert to coordinates)
        interpolated_lsm_data = self.interpolated_data_for_lsm(self.sample_lsm, interpolator_setup=interpolator_setup)
        interpolated_lsm_data_mask = ~np.isnan(interpolated_lsm_data)
        interpolated_lsm_values = interpolated_lsm_data[interpolated_lsm_data_mask]
        interpolated_lsm_indices = np.array(np.where(interpolated_lsm_data_mask)).T
        interpolated_lsm_points = self.sample_lsm.map_indices_to_coordinates(interpolated_lsm_indices)
        interpolated_lsm_points_and_values = np.concatenate([interpolated_lsm_points, interpolated_lsm_values[:,np.newaxis]], axis=1)

        ## interpolate
        interpolated_data = measurements.util.interpolate.periodic_with_coordinates(interpolated_lsm_points_and_values, interpolation_points, self.sample_lsm, scaling_values=self.scaling_values, interpolator_setup=(2/min([self.sample_lsm.t_dim,self.sample_lsm.x_dim]),0,0,0))
        
        ## return
        assert np.all(np.isfinite(interpolated_data))
        return interpolated_data
   

    def interpolated_data_for_points(self, interpolator_setup=None):
        interpolator_setup = self.get_interpolator_setup(interpolator_setup)
        filename = self.interpolated_data_filename.format(points='lexsorted_points', interpolator_setup=self._tuple_to_str(interpolator_setup), scaling_values=self._tuple_to_str(self.scaling_values))
        function = lambda :self.interpolated_data_for_points_calculate(interpolator_setup=interpolator_setup)
        return self.cache.get_value(filename, function)
        

