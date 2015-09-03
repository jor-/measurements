import bisect
import os
import numpy as np

import measurements.land_sea_mask.depth

import util.cache
import util.petsc.universal

import util.logging
logger = util.logging.logger


class LandSeaMask():

    def __init__(self, lsm, depth_level, t_dim=None, t_centered=True):
        assert lsm.ndim == 2
        assert len(depth_level) > 0

        self._lsm = lsm
        self._depth_level = np.asanyarray(depth_level)

        self._t_dim = t_dim
        self.t_centered = t_centered
        
        self.memory_cache = util.cache.MemoryCache()


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
        if len(key) >= 4:
            raise ValueError('Length of key has to be less or equal 3, but key is {}.'.format(key))
        return self._lsm[key[-2:]]



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

    def sea_indices_calculate(self):
        masked_map = self.masked_map(dtype=np.float16)
        sea_indices = np.array(np.where(np.logical_not(np.isnan(masked_map)))).transpose()
        logger.debug('Found {} sea indices in {}.'.format(sea_indices.shape[0], self))
        assert sea_indices.ndim == 2
        return sea_indices

    @property
    def sea_indices(self):
        return self.memory_cache.get_value('sea_indices_{}'.format(self.dim), self.sea_indices_calculate)
    

    def sea_coordinates_calculate(self):
        sea_coordinates = self.map_indices_to_coordinates(self.sea_indices)
        assert sea_coordinates.ndim == 2
        return sea_coordinates
    
    @property
    def sea_coordinates(self):
        return self.memory_cache.get_value('sea_coordinates_{}'.format(self.dim), self.sea_coordinates_calculate)
    

    def is_point_near_water(self, point, max_land_boxes=0):
        old_t_dim =  self.t_dim
        self.t_dim = None
        
        sea_indices = self.sea_indices
        map_index = np.asarray(self.coordinate_to_map_index(*point, discard_year=True))
        distance = np.abs(sea_indices - map_index[np.newaxis, :])
        
        self.t_dim = old_t_dim
        
        is_near_water = np.any(np.all(distance <= 0.5 + max_land_boxes, axis=1))
        logger.debug('Point {} is near water {} with max_nland_boxes {}.'.format(point, is_near_water, max_land_boxes))
        return is_near_water


    def points_near_water_mask(self, points, max_land_boxes=0):
        n = len(points)
        results = np.ones(n, dtype=np.bool)
        for i in range(n):
            results[i] = self.is_point_near_water(points[i], max_land_boxes=max_land_boxes)
        return results

    

    ## convert map indices and coordinates

    def coordinate_to_map_index(self, t, x, y, z, discard_year=True):
        ## t (center of the box, wrap around)
        try:
            t_dim = self.t_dim_with_value_check
        except ValueError:
            t_dim = None
        
        if t_dim is not None:
            if discard_year:
                t = (t % 1)
            ti = t * t_dim
            if self.t_centered:
                ti -= 0.5

        ## x (center of the box, wrap around)
        xi = (x % 360) / 360 * self.x_dim - 0.5

        ## y (center of the box, no wrap around)#, consider lowmost and topmost box)
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

        ## return
        if t_dim is not None:
            return (ti, xi, yi, zi)
        else:
            return (xi, yi, zi)


    def coordinates_to_map_indices(self, points, discard_year=True):
        result_ndim = points.ndim
        if points.ndim == 1:
            points = points[np.newaxis]
        logger.debug('Transforming {} coordinates to map indices for {} with discard year {}.'.format(len(points), self, discard_year))

        n = len(points)
        new_points = np.empty((n, self.ndim))
        for i in range(n):
            new_points[i] = self.coordinate_to_map_index(*points[i], discard_year=discard_year)

        if result_ndim == 1:
            new_points = new_points[0]

        logger.debug('Transforming from coordinates to map indices done.')
        return new_points


    def map_index_to_coordinate(self, ti, xi, yi, zi):
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
        x = ((xi + 0.5) % self.x_dim) / self.x_dim * 360

        ## y (center of the box, no wrap around, consider lowmost and topmost box)
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
            if zi_floor < 0.5:
                z = 2 * zi_fraction * (r[zi_floor] - c[zi_floor]) + c[zi_floor]
            else:
                zi_fraction -= 0.5
                z = 2 * zi_fraction * (c[zi_floor + 1] - r[zi_floor]) + r[zi_floor]
            assert z >= c[zi_floor] and z <= c[zi_floor+1]
        
        ## return
        if t_dim is not None:
            return (t, x, y, z)
        else:
            return (x, y, z)
            


    def map_indices_to_coordinates(self, points):
        result_ndim = points.ndim
        if points.ndim == 1:
            points = points[np.newaxis]
        logger.debug('Transforming {} map indices from {} to coordinates'.format(len(points), self))

        n = len(points)
        new_points = np.empty((n, self.ndim))
        for i in range(n):
            new_points[i] = self.map_index_to_coordinate(*points[i])

        if result_ndim == 1:
            new_points = new_points[0]

        assert new_points[:,1].max() <= 360 and new_points[:,1].min() >= 0 or new_points[:,1].max() <= 180 and new_points[:,1].min() >= -180
        assert new_points[:,2].max() <= 90 and new_points[:,2].min() >= -90
        assert new_points[:,3].min() >= 0

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


    def masked_map(self, default_value=0, dtype=np.float64, land_value=np.nan):
        masked_map = np.ones(self.dim, dtype=dtype) * default_value
        self.apply_mask(masked_map, land_value=land_value)
        return masked_map


    def insert_coordinate_values_in_map(self, values, no_data_value=0, apply_mask_last=True):
        values = np.copy(values)
        values[:,:-1] = self.coordinates_to_map_indices(values[:,:-1])
        return self.insert_index_values_in_map(values, no_data_value=no_data_value, apply_mask_last=apply_mask_last)


    def insert_index_values_in_map(self, values, no_data_value=0, apply_mask_last=True):
        logger.debug('Inserting {} values in map with value {} for no data.'.format(len(values), no_data_value))

        if (self.t_dim is None and values.shape[1] != 4) or (self.t_dim is not None and values.shape[1] != 5):
            raise ValueError('Values have wrong shape: Second dimension have to be 4 or 5, but it is {}.'.format(values.shape[1]))

        ## init map
        value_map = self.masked_map(default_value=no_data_value, dtype=values.dtype)
        number_map = self.masked_map(default_value=0, dtype=np.int32, land_value=-1)

        ## insert values: sum and count for each box
        for row in values:
            index = tuple(row[:-1].astype(np.int))
            value = row[-1]
            try:
                value_map[index]
            except IndexError:
                raise ValueError('Index {} exceeds dimension {}.'.format(index, value_map.shape))

            if value_map[index] == no_data_value:
                value_map[index] = value
            else:
                value_map[index] = value_map[index] + value

            number_map[index] = number_map[index] + 1

        ## average
        mask = number_map > 1
        value_map[mask] = value_map[mask] / number_map[mask]

        ## apply mask
        if apply_mask_last:
            self.apply_mask(value_map)

        return value_map


    ## plot
    def plot(self):
        import util.plot
        file = '/tmp/{}.png'.format(self)
        util.plot.data(self.lsm, file, land_value=0, power_limits=(-10,10))




class LandSeaMaskFromFile(LandSeaMask):

    def __init__(self, lsm_dir, t_dim=None, t_centered=True):
        from .constants import LSM_NPY_FILENAME, DEPTH_NPY_FILENAME

        cache = util.cache.HDD_NPY_Cache(lsm_dir)
        depth = cache.get_value(DEPTH_NPY_FILENAME, self._calculate_depth)
        lsm = cache.get_value(LSM_NPY_FILENAME, self._calculate_lsm)

        super().__init__(lsm, depth, t_dim=t_dim, t_centered=t_centered)


    def _calculate_lsm(self):
        raise NotImplementedError

    def _calculate_depth(self):
        raise NotImplementedError



class LandSeaMaskTMM(LandSeaMaskFromFile):
    def __init__(self, t_dim=None, t_centered=True):
        from .constants import TMM_DIR
        super().__init__(TMM_DIR, t_dim=t_dim, t_centered=t_centered)


    def _calculate_lsm(self):
        from .constants import TMM_PETSC_FILE

        lsm = util.petsc.universal.load_petsc_mat_to_array(TMM_PETSC_FILE, dtype=np.int16)
        lsm = lsm.transpose() # metos3d: x and y are changed

        assert lsm.shape == (128, 64) and lsm.min() == 0 and lsm.max() == 15

        return lsm


    def _calculate_depth(self):
        from .constants import TMM_DEPTH_TXT_FILE

        ## read values from txt
        depth = np.genfromtxt(TMM_DEPTH_TXT_FILE, dtype=np.int16, comments='#', usecols=(0,))
        assert depth.ndim == 1 and depth.shape[0] == 16

        return depth


    def __str__(self):
        return super().__str__() + '_tmm'



class LandSeaMaskWOA13(LandSeaMaskFromFile):
    def __init__(self, t_dim=None, t_centered=True):
        from .constants import WOA13_DIR
        super().__init__(WOA13_DIR, t_dim=t_dim, t_centered=t_centered)


    def _calculate_lsm(self):
        from .constants import WOA13_LSM_TXT_FILE

        ## read values from txt with axis order: x y z
        lsm = np.genfromtxt(WOA13_LSM_TXT_FILE, dtype=float, delimiter=',', comments='#', usecols=(1, 0, 2))

        ## normalize values
#         x = lsm[:,0]
#         x[x < 0] += 360
#         lsm[:,0] = x
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


    def _calculate_depth(self):
        from .constants import WOA13_DEPTH_TXT_FILE

        ## read values from txt
        depth = np.genfromtxt(WOA13_DEPTH_TXT_FILE, dtype=np.int16, comments='#', usecols=(0,))
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



