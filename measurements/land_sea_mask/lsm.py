import bisect
import os

import numpy as np
import overrides

import util.cache.file_based
import util.cache.memory_based
import util.petsc.universal
import util.logging

import measurements.constants
import measurements.land_sea_mask.depth
import measurements.land_sea_mask.constants

logger = util.logging.logger



class LandSeaMask():

    def __init__(self, lsm, depth_level, t_dim=None, t_centered=True):
        assert lsm.ndim == 2
        assert len(depth_level) > 0

        self._lsm = lsm
        self._depth_level = np.asanyarray(depth_level)

        self._t_dim = t_dim
        self.t_centered = t_centered


    ## equal

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.dim == other.dim and self.t_centered == other.t_centered and np.all(self.z == other.z) and np.all(self.lsm == other.lsm)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented


    ## dims

    @property
    def t_dim(self):
        return self._t_dim

    @t_dim.setter
    def t_dim(self, new_t_dim):
        self._t_dim = new_t_dim

    @property
    def t_dim_with_value_check(self):
        t_dim = self.t_dim
        if t_dim is not None and t_dim != 0:
            return t_dim
        else:
            raise ValueError('T dim is not set.')
        self.sea_indices.cache_clear()
        self.sea_coordinates.cache_clear()


    @property
    def x_dim(self):
        return self._lsm.shape[0]

    @property
    def y_dim(self):
        return self._lsm.shape[1]

    @property
    def z_dim(self):
        return len(self._depth_level) - 1

    @property
    def space_dim(self):
        return (self.x_dim, self.y_dim, self.z_dim)

    @property
    def dim(self):
        if self.t_dim is None or self.t_dim == 0:
            return (self.x_dim, self.y_dim, self.z_dim)
        else:
            return (self.t_dim, self.x_dim, self.y_dim, self.z_dim)

    @property
    def ndim(self):
        return len(self.dim)


    ## z
    
    @property
    def z(self):
        return self._depth_level

    @z.setter
    def z(self, new_z_values):
        logger.debug('Regridding z from {} to {}.'.format(self.z, new_z_values))
        new_z_values = np.asanyarray(new_z_values)

        z_values = self.z
        for i in range(len(z_values)):
            self._lsm[self._lsm == i] = bisect.bisect_left(new_z_values, z_values[i])
        self._depth_level = new_z_values

    @property
    def z_left(self):
        return self.z[:-1]

    @property
    def z_right(self):
        return self.z[1:]

    @property
    def z_center(self):
        return (self.z_left + self.z_right) / 2

    @property
    def separation_values(self):
        return (1/self.t_dim_with_value_check, 360/self.x_dim, 180/self.y_dim, self.z)


    ## lsm

    @property
    def lsm(self):
        return self._lsm

    @property
    def shape(self):
        return self._lsm.shape

    def __getitem__(self, key):
        if len(key) == 2:
            return self._lsm[key]
        elif len(key) == 3:
            return self._lsm[key[1:]]
        elif len(key) == 4:
            return self._lsm[key[1:3]] > key[3]
        else:
            raise ValueError('Length of key has to be in (2, 3, 4), but key is {}.'.format(key))


    def __str__(self):
        try:
            t_dim = self.t_dim_with_value_check
        except ValueError:
            t_dim = None
        if t_dim is not None:
            return 'lsm_{}'.format(t_dim)
        else:
            return 'lsm'


    ## indices

    @property
    @util.cache.memory_based.decorator(dependency='self.t_dim')
    def sea_indices(self):
        sea_indices = np.array(np.where(self.bool_mask)).transpose()
        logger.debug('Found {} sea indices in {}.'.format(sea_indices.shape[0], self))
        assert sea_indices.ndim == 2
        return sea_indices


    @property
    @util.cache.memory_based.decorator(dependency='self.t_dim')
    def sea_coordinates(self):
        sea_coordinates = self.map_indices_to_coordinates(self.sea_indices)
        assert sea_coordinates.ndim == 2
        return sea_coordinates
    

    def is_coordinate_near_water(self, point, max_box_distance_to_water=0):
        old_t_dim =  self.t_dim
        self.t_dim = 0
        
        sea_indices = self.sea_indices
        map_index = np.asarray(self.coordinate_to_map_index(*point, discard_year=True, int_indices=True))
        distance = np.abs(sea_indices - map_index[np.newaxis, :])
        
        self.t_dim = old_t_dim
        
        is_near_water = np.any(np.all(distance <= max_box_distance_to_water, axis=1))
        logger.debug('Coordinate {} is near water {} with max_box_distance_to_water {}.'.format(point, is_near_water, max_box_distance_to_water))
        return is_near_water


    def coordinates_near_water_mask(self, points, max_box_distance_to_water=0):
        n = len(points)
        results = np.empty(n, dtype=np.bool)
        for i in range(n):
            results[i] = self.is_coordinate_near_water(points[i], max_box_distance_to_water=max_box_distance_to_water)
        return results


    def box_bounds_of_map_index(self, map_index):
        map_index = np.asarray(map_index)
        t_centered = self.t_centered and len(map_index) >= 4
        
        map_index = map_index + 0.5
        if not t_centered:
            map_index[0] = map_index[0] -  0.5
        
        lower_bound = np.floor(map_index)
        lower_bound = lower_bound - 0.5
        if not t_centered:
            lower_bound[0] = lower_bound[0] +  0.5
        
        box_bounds = np.array([lower_bound, lower_bound + 1]) 
        box_bounds = box_bounds.transpose()
        
        assert len(box_bounds) == len(map_index)     
        assert box_bounds.shape[1] == 2
        assert np.all(box_bounds[:,1] - box_bounds[:,0] == 1)
        return box_bounds


    def box_bounds_of_map_indices(self, map_indices):
        ## prepare input
        result_ndim = map_indices.ndim
        if map_indices.ndim == 1:
            map_indices = map_indices[np.newaxis]

        ## calculate
        n = len(map_indices)
        logger.debug('Transforming {} map indices to box bounds for {}.'.format(n, self))

        box_bounds = np.empty(map_indices.shape + (2,))
        for i in range(n):
            box_bounds[i] = self.box_bounds_of_map_index(map_indices[i])

        ## return
        if result_ndim == 1:
            box_bounds = box_bounds[0]

        logger.debug('Transforming map indices to box bounds done.')
        return box_bounds


    @property
    @util.cache.memory_based.decorator(dependency='self.t_dim')
    def number_of_map_indices(self):
        t_dim = self.t_dim
        if t_dim is None:
            t_dim = 1
        return self.lsm.sum() * self.t_dim


    ## volume
    
    @staticmethod
    def volume_of_coordinate_box(bounds):
        bounds = np.asanyarray(bounds)
        assert bounds.ndim == 2
        assert bounds.shape[0] >= 3
        assert bounds.shape[1] == 2
        # assert bounds.shape == (3, 2)
        
        bounds = bounds[-3:]
        assert np.all(bounds[1] >= -90)
        assert np.all(bounds[1] <= 90)
        assert np.all(bounds[2] >= 0)
        
        alpha = bounds[0]
        beta = bounds[1]
        r = measurements.constants.EARTH_RADIUS - bounds[2]
        r = r[::-1]
        
        s = np.pi / (2 * 90)
        v = 1/3 * s * (alpha[1] - alpha[0]) *  (r[1]**3 - r[0]**3) * (np.sin(s * beta[1]) - np.sin(s * beta[0]))
        assert v >= 0
        return v
    
    
    @staticmethod
    def volume_of_coordinate_boxes(bounds):
        ## prepare input
        bounds = np.asanyarray(bounds)
        
        result_ndim = bounds.ndim
        if bounds.ndim == 2:
            bounds = bounds[np.newaxis]
        
        assert bounds.ndim == 3
        assert bounds.shape[1] >= 3
        assert bounds.shape[2] == 2
        
        ## calculate
        n = len(bounds)
        logger.debug('Calculating volume of {} coordinate boxes.'.format(n))
        
        box_volumes = np.empty(n)
        for i in range(n):
            box_volumes[i] = LandSeaMask.volume_of_coordinate_box(bounds[i])

        ## return
        if result_ndim == 1:
            box_volumes = box_volumes[0]

        logger.debug('Volume of coordinate boxes are calculated.')
        return box_volumes
    
    
    def volume_of_boxes_of_map_indices(self, map_indices):
        logger.debug('Calculating volume of boxes of {} map indices.'.format(len(map_indices)))
        
        ## calculate box bounds as map indices
        box_bounds = self.box_bounds_of_map_indices(map_indices)
        assert box_bounds.shape[2] == 2
        assert np.all(box_bounds[:,:,1] > box_bounds[:,:,0])

        ## calculate box bounds as coordinates
        for i in range(box_bounds.shape[2]):
           box_bounds[:,:,i] = self.map_indices_to_coordinates(box_bounds[:,:,i], use_modulo_for_x=False)
        assert np.all(box_bounds[:,:,1] > box_bounds[:,:,0])
        
        ## calculate volumes
        volumes = self.volume_of_coordinate_boxes(box_bounds)
        
        ## return
        return volumes
    
    
    @property
    def volume_map(self):
        logger.debug('Calculating volume map.')
        
        ## calculate sea map indices
        sea_indices = self.sea_indices
        
        ## calculate volume map
        volumes = self.volume_of_boxes_of_map_indices(sea_indices)
        volumes_with_indices = np.concatenate([sea_indices, volumes[:,np.newaxis]], axis=1)
        volume_map = self.insert_index_values_in_map(volumes_with_indices, no_data_value=np.inf)
        
        ## return
        return volume_map
    
    
    @property
    def normalized_volume_weights_map(self):
        volume_map = self.volume_map
        normalized_volume_map = volume_map / np.nansum(volume_map)
        assert np.isclose(np.nansum(normalized_volume_map), 1)
        return normalized_volume_map


    ## convert map indices and coordinates

    def coordinate_to_map_index(self, t, x, y, z, discard_year=False, int_indices=True):
        ## t (center of the box, wrap around)
        try:
            t_dim = self.t_dim_with_value_check
        except ValueError:
            t_dim = None
        
        if t_dim is not None:
            ti = t * t_dim
            if discard_year:
                ti = (ti % t_dim)
            if self.t_centered:
                ti -= 0.5

        ## x (center of the box, wrap around)
        xi = (x % 360) / 360 * self.x_dim - 0.5

        ## y (center of the box, no wrap around)
        yi = (y + 90) / 180 * self.y_dim  - 0.5

        ## z (center of the box, no wrap around)
        r = self.z_right
        c = self.z_center
        m = len(c)-1
        
        zi = bisect.bisect_left(c, z) - 1
        if zi == -1:
            assert z <= c[0]
            zi = 1/2 * (z / c[0] - 1)
        elif zi == m:
            assert z >= c[m]
            zi = m +  1/2 * (z - c[m]) / (r[m] - c[m])
        else:
            assert z >= c[zi] and z <= c[zi+1]
            zi += 1/2 * (min(z, r[zi]) - c[zi]) / (r[zi] - c[zi]) + 1/2 * (max(z, r[zi]) - r[zi]) / (c[zi+1] - r[zi])

        ## concatenate (float) index
        if t_dim is not None:
            map_index = (ti, xi, yi, zi)
        else:
            map_index = (xi, yi, zi)
        
        ## convert to int if needed
        if int_indices:
            ## round half up (and not round half to even which is numpys default)
            map_index = np.array(map_index)
            tie_mask = map_index % 1 == 0.5
            map_index[tie_mask] = map_index[tie_mask] + 0.5
            map_index = np.array(np.round(map_index), dtype=np.int32)
            ## check bounds
            if map_index[-1] > self.z_dim:          # below z bottom
                map_index[-1] = self.z_dim  
            if map_index[-2] == self.y_dim:         # y = 90 degree
                map_index[-2] = self.y_dim - 1
            ## convert back to tuple
            map_index = tuple(map_index)
            assert len(map_index) == 3 or (not discard_year) or (map_index[0] >= 0 and map_index[0] < self.t_dim)
            assert (map_index[-3] >= 0 and map_index[-3] < self.x_dim)
            assert (map_index[-2] >= 0 and map_index[-2] < self.y_dim)
            assert (map_index[-1] >= 0 and map_index[-1] <= self.z_dim)
        
        ## return
        return map_index


    def coordinates_to_map_indices(self, points, discard_year=False, int_indices=True):
        points = np.asanyarray(points)
        result_ndim = points.ndim
        if points.ndim == 1:
            points = points[np.newaxis]
        logger.debug('Transforming {} coordinates to map indices for {} with discard year {} and int_indices {}.'.format(len(points), self, discard_year, int_indices))

        n = len(points)
        if int_indices:
            dtype= np.int32
        else:
            dtype= np.float
        new_points = np.empty((n, self.ndim), dtype=dtype)
        
        for i in range(n):
            new_points[i] = self.coordinate_to_map_index(*points[i], discard_year=discard_year, int_indices=int_indices)

        if result_ndim == 1:
            new_points = new_points[0]

        logger.debug('Transforming from coordinates to map indices done.')
        return new_points


    def map_index_to_coordinate(self, ti, xi, yi, zi, use_modulo_for_x=True):
        ## t (left or center of the box, wrap around)
        try:
            t_dim = self.t_dim_with_value_check
        except ValueError:
            t_dim = None
        if t_dim is not None:
            if self.t_centered:
                ti += 0.5
            t = ti / self.t_dim_with_value_check

        ## x (center of the box, wrap around)
        x = xi + 0.5
        if use_modulo_for_x:
            x = x % self.x_dim
        x = x / self.x_dim * 360

        ## y (center of the box, no wrap around)
        y = (yi + 0.5) / self.y_dim * 180 - 90

        ## z (center of the box, no wrap around)
        r = self.z_right
        c = self.z_center
        m = len(c)-1
        
        if zi < 0:
            z = (2 * zi + 1) * c[0]
            assert z <= c[0]
        elif zi >= m:
            z = 2 * (zi - m) * (r[m] - c[m]) + c[m]
            assert z >= c[m]
        else:
            zi_floor = int(np.floor(zi))
            zi_fraction = zi % 1
            if zi_fraction < 0.5:
                z = 2 * zi_fraction * (r[zi_floor] - c[zi_floor]) + c[zi_floor]
            else:
                zi_fraction -= 0.5
                z = 2 * zi_fraction * (c[zi_floor + 1] - r[zi_floor]) + r[zi_floor]
            assert z >= c[zi_floor] and z <= c[zi_floor+1]
        
        ## concatenate coordinates
        if t_dim is not None:
            coordinates = (t, x, y, z)
        else:
            coordinates = (x, y, z)
        
        ## return
        assert not use_modulo_for_x or (coordinates[-3] >= 0 and coordinates[-3] <= 360)
        assert coordinates[-2] >= -90
        assert coordinates[-2] <= 90
        assert coordinates[-1] >= 0
        assert coordinates[-1] <= measurements.constants.MAX_SEA_DEPTH
        return coordinates
            

    def map_indices_to_coordinates(self, points, use_modulo_for_x=True):
        result_ndim = points.ndim
        if points.ndim == 1:
            points = points[np.newaxis]
        logger.debug('Transforming {} map indices from {} to coordinates with use_modulo_for_x {}'.format(len(points), self, use_modulo_for_x))

        n = len(points)
        new_points = np.empty((n, self.ndim))
        for i in range(n):
            new_points[i] = self.map_index_to_coordinate(*points[i], use_modulo_for_x=use_modulo_for_x)

        if result_ndim == 1:
            new_points = new_points[0]

        logger.debug('Transforming from map indices to coordinates done.')
        return new_points


    ## values to map

    def apply_mask(self, array, land_value=np.nan):
        if self.dim != array.shape:
            raise ValueError('Array must have the same dims as lsm, but its shape is {} and it has to be {}.'.format(array.shape, self.dim))

        for i in np.ndindex(self.dim[:-1]):
            z_max = self[i]
            array[i][z_max:] = land_value
        return array


    def masked_map(self, default_value=0, land_value=np.nan, dtype=np.float64):
        masked_map = np.ones(self.dim, dtype=dtype) * default_value
        self.apply_mask(masked_map, land_value=land_value)
        return masked_map

    
    @property
    def bool_mask(self):
        return self.masked_map(dtype=np.bool, default_value=True, land_value=False)


    def insert_coordinate_values_in_map(self, values, no_data_value=0, skip_values_on_land=True):
        values = np.copy(values)
        values[:,:-1] = self.coordinates_to_map_indices(values[:,:-1], discard_year=True, int_indices=True)
        return self.insert_index_values_in_map(values, no_data_value=no_data_value, skip_values_on_land=skip_values_on_land)


    def insert_index_values_in_map(self, values, no_data_value=0, skip_values_on_land=True):
        logger.debug('Inserting {} values in map with value {} for no data.'.format(len(values), no_data_value))

        if values.shape[1] not in (4, 5):
            raise ValueError('Values have wrong shape: Second dimension have to be 4 or 5, but it is {}.'.format(values.shape[1]))
        if np.isnan(no_data_value):
            raise ValueError('No data value can not be NAN.')
        
        ## remove time dim if values have no time
        if values.shape[1] == 4:
            old_t_dim = self.t_dim
            self.t_dim = None

        ## init map
        value_map = self.masked_map(default_value=no_data_value, land_value=np.nan, dtype=values.dtype)
        number_map = self.masked_map(default_value=0, land_value=0, dtype=np.int32)

        ## insert values: sum and count for each box
        for row in values:
            index = tuple(row[:-1].astype(np.int))
            value = row[-1]
            try:
                value_at_index = value_map[index]
            except IndexError:
                raise ValueError('Index {} exceeds dimension {}.'.format(index, value_map.shape))

            if not skip_values_on_land or not np.isnan(value_at_index):
                if value_at_index == no_data_value or np.isnan(value_at_index):
                    value_map[index] = value
                else:
                    value_map[index] = value_at_index + value
                number_map[index] = number_map[index] + 1

        ## average
        mask = number_map > 1
        value_map[mask] = value_map[mask] / number_map[mask]

        ## restore time dim
        if values.shape[1] == 4:
            self.t_dim = old_t_dim
    
        ## return
        return value_map


    ## plot
    def plot(self):
        import util.plot
        file = '/tmp/{}.png'.format(self)
        util.plot.data(self.lsm, file, land_value=0, power_limit=10)
    
    
    ## copy
    def copy(self):
        import copy
        return copy.deepcopy(self)



class LandSeaMaskFromFile(LandSeaMask):

    def __init__(self, lsm_dir, t_dim=None, t_centered=True):
        self._lsm_file = os.path.join(lsm_dir, measurements.land_sea_mask.constants.LSM_NPY_FILENAME)
        self._depth_file = os.path.join(lsm_dir, measurements.land_sea_mask.constants.DEPTH_NPY_FILENAME)
        
        depth = self._calculate_depth()
        lsm = self._calculate_lsm()
        
        super().__init__(lsm, depth, t_dim=t_dim, t_centered=t_centered)


    @util.cache.file_based.decorator(cache_file_function=lambda self: self._lsm_file)
    def _calculate_lsm(self):
        raise NotImplementedError

    @util.cache.file_based.decorator(cache_file_function=lambda self: self._depth_file)
    def _calculate_depth(self):
        raise NotImplementedError



class LandSeaMaskTMM(LandSeaMaskFromFile):
    
    def __init__(self, t_dim=None, t_centered=True):
        super().__init__(measurements.land_sea_mask.constants.TMM_DIR, t_dim=t_dim, t_centered=t_centered)


    @util.cache.file_based.decorator(cache_file_function=lambda self: self._lsm_file)
    @overrides.overrides
    def _calculate_lsm(self):
        lsm = util.petsc.universal.load_petsc_mat_to_array(measurements.land_sea_mask.constants.TMM_PETSC_FILE, dtype=np.int16)
        lsm = lsm.transpose() # metos3d: x and y are changed
        assert lsm.shape == (128, 64) and lsm.min() == 0 and lsm.max() == 15
        return lsm


    @util.cache.file_based.decorator(cache_file_function=lambda self: self._depth_file)
    @overrides.overrides
    def _calculate_depth(self):
        ## read values from txt
        depth = np.genfromtxt(measurements.land_sea_mask.constants.TMM_DEPTH_TXT_FILE, dtype=np.int16, comments='#', usecols=(0,))
        assert depth.ndim == 1 and depth.shape[0] == 16
        return depth


    def __str__(self):
        return super().__str__() + '_tmm'



class LandSeaMaskWOA13(LandSeaMaskFromFile):
    
    def __init__(self, t_dim=None, t_centered=True):
        super().__init__(measurements.land_sea_mask.constants.WOA13_DIR, t_dim=t_dim, t_centered=t_centered)


    @util.cache.file_based.decorator(cache_file_function=lambda self: self._lsm_file)
    @overrides.overrides
    def _calculate_lsm(self):
        ## read values from txt with axis order: x y z
        lsm = np.genfromtxt(measurements.land_sea_mask.constants.WOA13_LSM_TXT_FILE, dtype=float, delimiter=',', comments='#', usecols=(1, 0, 2))

        ## normalize values
        lsm[:,0] = lsm[:,0] % 360

        lsm = lsm - lsm.min(axis=0)

        ## convert to int
        lsm_int = lsm.astype(np.int16)

        assert np.all(lsm_int == lsm)
        assert lsm_int[:,0].min() == 0 and lsm_int[:,0].max() == 359 and lsm_int[:,1].min() == 0 and lsm_int[:,1].max() == 179 and lsm_int[:,2].min() == 0 and lsm_int[:,2].max() == 137

        ## convert in 2 dim
        lsm = np.empty((360, 180), dtype=np.int16)
        for x, y, z in lsm_int:
            lsm[x, y] = z
        
        assert lsm.min() == 0 and lsm.max() == 137
        return lsm


    @util.cache.file_based.decorator(cache_file_function=lambda self: self._depth_file)
    @overrides.overrides
    def _calculate_depth(self):
        ## read values from txt
        depth = np.genfromtxt(measurements.land_sea_mask.constants.WOA13_DEPTH_TXT_FILE, dtype=np.int16, comments='#', usecols=(0,))
        assert depth.ndim == 1 and depth.shape[0] == 138
        return depth


    def __str__(self):
        return super().__str__() + '_woa13'



class LandSeaMaskWOA13R(LandSeaMask):

    def __init__(self, t_dim=None, t_centered=True):
        depth = measurements.land_sea_mask.depth.values_TMM(max_value=5200, increment_step=2)
        depth.extend([6000, 8000, 10000])

        lsm_woa13 = LandSeaMaskWOA13()
        lsm_woa13.z = depth
        lsm = lsm_woa13.lsm

        super().__init__(lsm, depth, t_dim=t_dim, t_centered=t_centered)

    def __str__(self):
        return super().__str__() + '_woa13r'



