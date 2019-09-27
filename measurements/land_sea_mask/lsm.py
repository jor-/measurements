import bisect
import os

import numpy as np
import overrides

import util.cache.file
import util.cache.memory
import util.petsc.universal
import util.logging

import measurements.constants
import measurements.land_sea_mask.depth
import measurements.land_sea_mask.constants


class LandSeaMask():

    def __init__(self, lsm, depth_level, t_dim=None, t_centered=True):
        assert lsm.ndim == 2
        assert len(depth_level) > 0

        self._lsm = lsm
        self._depth_level = np.asanyarray(depth_level)

        self._t_dim = t_dim
        self.t_centered = t_centered

    # equal

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.dim == other.dim and self.t_centered == other.t_centered and np.all(self.z == other.z) and np.all(self.lsm == other.lsm)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    # dims

    @property
    def t_dim(self):
        t_dim = self._t_dim
        if t_dim is not None and t_dim != 0:
            return t_dim
        else:
            raise ValueError('T dim is not set.')

    @t_dim.setter
    def t_dim(self, new_t_dim):
        self._t_dim = new_t_dim

    @property
    def x_dim(self):
        return self._lsm.shape[0]

    @property
    def y_dim(self):
        return self._lsm.shape[1]

    @property
    def z_dim(self):
        return len(self.z) - 1

    @property
    def space_dim(self):
        return (self.x_dim, self.y_dim, self.z_dim)

    @property
    def dim(self):
        try:
            t_dim = self.t_dim
        except ValueError:
            dim = (self.x_dim, self.y_dim, self.z_dim)
        else:
            dim = (t_dim, self.x_dim, self.y_dim, self.z_dim)
        return dim

    @property
    def ndim(self):
        return len(self.dim)

    # z

    @property
    def z(self):
        return self._depth_level

    @z.setter
    def z(self, new_z_values):
        util.logging.debug('Regridding z from {} to {}.'.format(self.z, new_z_values))
        new_z_values = np.asanyarray(new_z_values)

        old_z_values = self.z
        for i in range(len(old_z_values)):
            self._lsm[self._lsm == i] = bisect.bisect_left(new_z_values, old_z_values[i])
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
        return (1 / self.t_dim, 360 / self.x_dim, 180 / self.y_dim, self.z)

    # lsm

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

    @property
    def name(self):
        try:
            t_dim = self.t_dim
        except ValueError:
            return 'lsm'
        else:
            return 'lsm_{}'.format(t_dim)

    def __str__(self):
        return self.name

    # indices

    @property
    @util.cache.memory.method_decorator(dependency='self._t_dim')
    def sea_indices(self):
        sea_indices = np.array(np.where(self.bool_mask())).transpose()
        util.logging.debug('Found {} sea indices in {}.'.format(sea_indices.shape[0], self))
        assert sea_indices.ndim == 2
        return sea_indices

    @property
    @util.cache.memory.method_decorator(dependency='self._t_dim')
    def sea_coordinates(self):
        sea_coordinates = self.map_indices_to_coordinates(self.sea_indices)
        assert sea_coordinates.ndim == 2
        return sea_coordinates

    def is_coordinate_near_water(self, point, max_box_distance_to_water=0):
        if max_box_distance_to_water is None or max_box_distance_to_water == float('inf'):
            is_near_water = True
        else:
            # check and convert max_box_distance_to_water
            try:
                max_box_distance_to_water = int(max_box_distance_to_water)
            except TypeError:
                raise ValueError('max_box_distance_to_water must be a non-negative integer or inf or None but it is {}.'.format(max_box_distance_to_water))
            if max_box_distance_to_water < 0:
                raise ValueError('max_box_distance_to_water must be a non-negative integer but it is {}.'.format(max_box_distance_to_water))

            # calculate and check distance
            old_t_dim = self._t_dim
            self.t_dim = 0
            sea_indices = self.sea_indices
            map_index = np.asarray(self.coordinate_to_map_index(*point, discard_year=True, int_indices=True))
            distance = np.abs(sea_indices - map_index[np.newaxis, :])
            self.t_dim = old_t_dim
            is_near_water = np.any(np.all(distance <= max_box_distance_to_water, axis=1))

        util.logging.debug('Coordinate {} is near water {} with max_box_distance_to_water {}.'.format(point, is_near_water, max_box_distance_to_water))
        return is_near_water

    def coordinates_near_water_mask(self, points, max_box_distance_to_water=0):
        n = len(points)
        if max_box_distance_to_water is None or max_box_distance_to_water == float('inf'):
            results = np.ones(n, dtype=np.bool)
        else:
            results = np.empty(n, dtype=np.bool)
            for i in range(n):
                results[i] = self.is_coordinate_near_water(points[i], max_box_distance_to_water=max_box_distance_to_water)
        return results

    def box_bounds_of_map_index(self, map_index):
        map_index = np.asarray(map_index)
        t_centered = self.t_centered and len(map_index) >= 4

        map_index = map_index + 0.5
        if not t_centered:
            map_index[0] = map_index[0] - 0.5

        lower_bound = np.floor(map_index)
        lower_bound = lower_bound - 0.5
        if not t_centered:
            lower_bound[0] = lower_bound[0] + 0.5

        box_bounds = np.array([lower_bound, lower_bound + 1])
        box_bounds = box_bounds.transpose()

        assert len(box_bounds) == len(map_index)
        assert box_bounds.shape[1] == 2
        assert np.all(box_bounds[:, 1] - box_bounds[:, 0] == 1)
        return box_bounds

    def box_bounds_of_map_indices(self, map_indices):
        # prepare input
        result_ndim = map_indices.ndim
        if map_indices.ndim == 1:
            map_indices = map_indices[np.newaxis]

        # calculate
        n = len(map_indices)
        util.logging.debug('Transforming {} map indices to box bounds for {}.'.format(n, self))

        box_bounds = np.empty(map_indices.shape + (2,))
        for i in range(n):
            box_bounds[i] = self.box_bounds_of_map_index(map_indices[i])

        # return
        if result_ndim == 1:
            box_bounds = box_bounds[0]

        util.logging.debug('Transforming map indices to box bounds done.')
        return box_bounds

    @property
    @util.cache.memory.method_decorator(dependency='self._t_dim')
    def number_of_map_indices(self):
        try:
            t_dim = self.t_dim
        except ValueError:
            t_dim = 1
        return self.lsm.sum() * self.t_dim

    # volume

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
        v = 1 / 3 * s * (alpha[1] - alpha[0]) * (r[1]**3 - r[0]**3) * (np.sin(s * beta[1]) - np.sin(s * beta[0]))
        assert v >= 0
        return v

    @staticmethod
    def volume_of_coordinate_boxes(bounds, dtype=np.float64):
        # prepare input
        bounds = np.asanyarray(bounds)

        result_ndim = bounds.ndim
        if bounds.ndim == 2:
            bounds = bounds[np.newaxis]

        assert bounds.ndim == 3
        assert bounds.shape[1] >= 3
        assert bounds.shape[2] == 2

        # calculate
        n = len(bounds)
        util.logging.debug('Calculating volume of {} coordinate boxes.'.format(n))

        box_volumes = np.empty(n, dtype)
        for i in range(n):
            box_volumes[i] = LandSeaMask.volume_of_coordinate_box(bounds[i])

        # return
        if result_ndim == 1:
            box_volumes = box_volumes[0]

        util.logging.debug('Volume of coordinate boxes are calculated.')
        return box_volumes

    def volume_of_boxes_of_map_indices(self, map_indices, dtype=np.float64):
        util.logging.debug('Calculating volume of boxes of {} map indices.'.format(len(map_indices)))

        # calculate box bounds as map indices
        box_bounds = self.box_bounds_of_map_indices(map_indices)
        assert box_bounds.shape[2] == 2
        assert np.all(box_bounds[:, :, 1] > box_bounds[:, :, 0])

        # calculate box bounds as coordinates
        for i in range(box_bounds.shape[2]):
            box_bounds[:, :, i] = self.map_indices_to_coordinates(box_bounds[:, :, i], use_modulo_for_x=False)
        assert np.all(box_bounds[:, :, 1] > box_bounds[:, :, 0])

        # calculate volumes
        volumes = self.volume_of_coordinate_boxes(box_bounds, dtype=dtype)

        # return
        return volumes

    def volumes_map(self, t_dim='default', dtype=np.float64):
        util.logging.debug(f'Calculating volume map of t_dim {t_dim} with dtype {dtype}.')
        # use defualt t_dim
        if t_dim == 'default':
            t_dim = self._t_dim
        # calculate without t_dim
        if t_dim is None or t_dim == 0:
            # save t_dim
            old_t_dim = self._t_dim
            self.t_dim = None
            # calculate volume map without t_dim
            sea_indices = self.sea_indices
            volumes = self.volume_of_boxes_of_map_indices(sea_indices, dtype=dtype)
            volumes_with_indices = np.concatenate([sea_indices, volumes[:, np.newaxis]], axis=1)
            volumes_map = self.insert_index_values_in_map(volumes_with_indices, no_data_value=np.inf)
            assert volumes_map.shape == self.dim[-3:]
            # restore t_dim
            self.t_dim = old_t_dim
        # calculate with t_dim
        else:
            volumes_map = self.volumes_map(t_dim=None, dtype=dtype)
            result_shape = (t_dim, *self.dim[-3:])
            volumes_map = np.broadcast_to(volumes_map, result_shape)
            assert volumes_map.ndim == 4
            assert volumes_map.shape[0] == t_dim
            assert volumes_map.shape[-3:] == self.dim[-3:]
        # return
        return volumes_map

    def normalized_volume_weights_map(self, t_dim='default', dtype=np.float64):
        util.logging.debug(f'Calculating normalized volume weight map of t_dim {t_dim} with dtype {dtype}.')
        # use defualt t_dim
        if t_dim == 'default':
            t_dim = self._t_dim
        # calculate without t_dim
        if t_dim is None or t_dim == 0:
            volume_map = self.volumes_map(t_dim=t_dim, dtype=dtype)
            normalized_volume_weights_map = volume_map / np.nansum(volume_map, dtype=dtype)
            assert normalized_volume_weights_map.shape == self.dim[-3:]
        # calculate with t_dim
        else:
            normalized_volume_weights_map = self.normalized_volume_weights_map(t_dim=None, dtype=dtype)
            result_shape = (t_dim, *self.dim[-3:])
            normalized_volume_weights_map = np.broadcast_to(normalized_volume_weights_map, result_shape)
            normalized_volume_weights_map = normalized_volume_weights_map / normalized_volume_weights_map.shape[0]
            assert normalized_volume_weights_map.ndim == 4
            assert normalized_volume_weights_map.shape[0] == t_dim
            assert normalized_volume_weights_map.shape[-3:] == self.dim[-3:]
        # return
        assert np.isclose(np.nansum(normalized_volume_weights_map, dtype=np.float128), 1)
        return normalized_volume_weights_map

    # convert map indices and coordinates

    def _float_index_to_int_index(self, f, dtype=np.int):
        f = np.asanyarray(f)
        # if 0.5 is fractional part round up (and not round half to even which is numpys default)
        mask = f % 1 == 0.5
        f[mask] = f[mask] + 0.5
        # convert to next integer
        f = np.round(f)
        # convert to integer array
        f = f.astype(dtype, casting='unsafe', copy=False)
        return f

    def t_to_map_index(self, t, discard_year=False, int_indices=True):
        # t (center of the box, wrap around)
        try:
            t_dim = self.t_dim
        except ValueError:
            ti = None
        else:
            ti = t * t_dim
            if discard_year:
                ti = ti % t_dim
            if self.t_centered:
                ti -= 0.5
            if int_indices:
                if discard_year:
                    dtype = np.min_scalar_type(t_dim - 1)
                else:
                    dtype = np.int
                ti = self._float_index_to_int_index(ti, dtype=dtype)
            assert ((not discard_year)
                    or ((int_indices or (not self.t_centered)) and np.all(np.logical_and(ti >= 0, ti < t_dim)))
                    or (((not int_indices) and self.t_centered) and np.all(np.logical_and(ti >= -0.5, ti < t_dim - 0.5))))
        # return
        return ti

    def x_to_map_index(self, x, int_indices=True):
        # x (center of the box, wrap around)
        x_dim = self.x_dim
        xi = (x % 360) / 360 * x_dim - 0.5
        # to int index
        if int_indices:
            xi = self._float_index_to_int_index(xi, dtype=np.min_scalar_type(x_dim - 1))
        # return
        assert (int_indices and np.all(np.logical_and(xi >= 0, xi < x_dim))
                or ((not int_indices) and np.all(np.logical_and(xi >= -0.5, xi < x_dim - 0.5))))
        return xi

    def y_to_map_index(self, y, int_indices=True):
        # y (center of the box, no wrap around)
        y_dim = self.y_dim
        yi = (y + 90) / 180 * y_dim - 0.5
        # to int index
        if int_indices:
            yi = self._float_index_to_int_index(yi, dtype=np.min_scalar_type(y_dim - 1))
            # case y is 90 degree
            yi = np.asanyarray(yi)
            yi[yi == y_dim] = y_dim - 1
        # return
        assert (int_indices and np.all(np.logical_and(yi >= 0, yi < y_dim))
                or ((not int_indices) and np.all(np.logical_and(yi >= -0.5, yi <= y_dim - 0.5))))
        return yi

    def z_to_map_index(self, z, int_indices=True):
        # z (center of the box, no wrap around)
        z_dim = self.z_dim
        r = self.z_right
        c = self.z_center
        assert len(c) == len(r)
        assert len(c) == z_dim

        # find index to the left
        z = np.asanyarray(z)
        zi = np.searchsorted(c, z, side='left')
        zi = np.asanyarray(zi)

        # masks
        mask_below = zi == 0
        mask_above = zi == z_dim
        mask_between = np.logical_and(np.logical_not(mask_below),
                                      np.logical_not(mask_above))

        # between centered values
        zi = zi - 1
        assert np.all(np.logical_and(z[mask_between] >= c[zi[mask_between]],
                                     z[mask_between] <= c[zi[mask_between] + 1]))
        zi_mask_between_offset = (+ 0.5 * (np.minimum(z[mask_between], r[zi[mask_between]]) - c[zi[mask_between]])
                                  / (r[zi[mask_between]] - c[zi[mask_between]])
                                  + 0.5 * (np.maximum(z[mask_between], r[zi[mask_between]]) - r[zi[mask_between]])
                                  / (c[zi[mask_between] + 1] - r[zi[mask_between]]))
        zi = np.asanyarray(zi, dtype=np.float)
        if len(zi_mask_between_offset) > 0:
            zi[mask_between] = zi[mask_between] + zi_mask_between_offset
        del mask_between

        # above first centered value
        assert np.all(z[mask_below] <= c[0])
        zi[mask_below] = 0.5 * (z[mask_below] / c[0] - 1)
        del mask_below

        # below last centered value
        assert np.all(z[mask_above] >= c[-1])
        zi[mask_above] = z_dim - 1 + 0.5 * (z[mask_above] - c[-1]) / (r[-1] - c[-1])
        del mask_above

        # to int index
        if int_indices:
            zi = self._float_index_to_int_index(zi, dtype=np.min_scalar_type(z_dim))
            zi[zi > z_dim] = z_dim

        # return
        assert ((int_indices
                 and np.all(np.logical_and(zi >= 0, zi <= z_dim))
                 and np.all(self.z[zi] <= z)
                 and np.all(z[zi < z_dim] < self.z[zi[zi < z_dim] + 1]))
                or ((not int_indices) and np.all(zi >= -0.5)))
        return zi

    def _coordinate_to_map_index_single_axis_function(self, axis, discard_year=False, int_indices=True):
        convert_functions = (lambda t: self.t_to_map_index(t, discard_year=discard_year, int_indices=int_indices),
                             lambda x: self.x_to_map_index(x, int_indices=int_indices),
                             lambda y: self.y_to_map_index(y, int_indices=int_indices),
                             lambda z: self.z_to_map_index(z, int_indices=int_indices))
        convert_function = convert_functions[axis]
        return convert_function

    def coordinates_to_map_indices_single_axis(self, values, axis, discard_year=False, int_indices=True):
        values = np.asanyarray(values)
        assert values.ndim == 1
        convert_function = self._coordinate_to_map_index_single_axis_function(axis, discard_year=discard_year, int_indices=int_indices)
        map_indices = convert_function(values)
        return map_indices

    def coordinates_to_map_indices(self, points, discard_year=False, int_indices=True):
        # convert points to 2 dim array
        points = np.asanyarray(points)
        result_ndim = points.ndim
        if result_ndim == 1:
            points = points[np.newaxis]
        util.logging.debug('Transforming {} coordinates to map indices for {} with discard year {} and int_indices {}.'.format(len(points), self, discard_year, int_indices))

        # create map indices array
        if int_indices:
            if discard_year or points.shape[1] < 4:
                dtype = np.min_scalar_type(np.max(self.dim) - 1)
            else:
                dtype = np.int
        else:
            dtype = np.float
        (n, m) = points.shape
        map_indices = np.empty((n, m), dtype=dtype)

        # convert
        for i in range(-1, -m - 1, -1):
            map_indices[:, i] = self.coordinates_to_map_indices_single_axis(points[:, i], i, discard_year=discard_year, int_indices=int_indices)

        # return
        if result_ndim == 1:
            map_indices = map_indices[0]
        util.logging.debug('Transforming from coordinates to map indices done.')
        return map_indices

    def coordinate_to_map_index(self, t, x, y, z, discard_year=False, int_indices=True):
        # convert to point
        try:
            t_dim = self.t_dim
        except ValueError:
            point = np.array((x, y, z))
        else:
            point = np.array((t, x, y, z))

        # return
        map_index = self.coordinates_to_map_indices(point, discard_year=discard_year, int_indices=int_indices)
        return map_index

    def map_index_to_coordinate(self, ti, xi, yi, zi, use_modulo_for_x=True):
        # t (left or center of the box, wrap around)
        try:
            t_dim = self.t_dim
        except ValueError:
            t_dim = None
        else:
            if self.t_centered:
                ti += 0.5
            t = ti / t_dim

        # x (center of the box, wrap around)
        x = xi + 0.5
        if use_modulo_for_x:
            x = x % self.x_dim
        x = x / self.x_dim * 360

        # y (center of the box, no wrap around)
        y = (yi + 0.5) / self.y_dim * 180 - 90

        # z (center of the box, no wrap around)
        r = self.z_right
        c = self.z_center
        m = len(c) - 1

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
            assert z >= c[zi_floor] and z <= c[zi_floor + 1]

        # concatenate coordinates
        if t_dim is not None:
            coordinates = (t, x, y, z)
        else:
            coordinates = (x, y, z)

        # return
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
        assert points.ndim == 2
        m = self.ndim
        assert points.shape[1] == m
        util.logging.debug('Transforming {} map indices from {} to coordinates with use_modulo_for_x {}'.format(len(points), self, use_modulo_for_x))

        n = len(points)
        new_points = np.empty((n, m))
        for i in range(n):
            points_i = points[i]
            if m == 3:
                points_i = (None, *points_i)
            new_points[i] = self.map_index_to_coordinate(*points_i, use_modulo_for_x=use_modulo_for_x)

        if result_ndim == 1:
            new_points = new_points[0]

        util.logging.debug('Transforming from map indices to coordinates done.')
        return new_points

    # values map

    def apply_mask(self, array, land_value=np.nan):
        if self.dim != array.shape:
            raise ValueError(f'Array must have the same dims as lsm, but its shape is {array.shape} and it has to be {self.dim}.')

        for i in np.ndindex(self.dim[:-1]):
            z_max = self[i]
            array[i][z_max:] = land_value
        return array

    def masked_map(self, default_value=0, land_value=np.nan, dtype=np.float64):
        masked_map = np.ones(self.dim, dtype=dtype) * default_value
        masked_map = self.apply_mask(masked_map, land_value=land_value)
        return masked_map

    def bool_mask(self):
        mask = self.masked_map(dtype=np.bool, default_value=True, land_value=False)
        return mask

    def apply_ranges_to_mask(self, mask, outside_value=np.nan, t_from=None, t_to=None, x_from=None, x_to=None, y_from=None, y_to=None, z_from=None, z_to=None):

        def set_outside_value_single_axis(axis, left_index=None, right_index=None):
            mask_index = (slice(None), ) * axis + (slice(left_index, right_index), )
            mask[mask_index] = outside_value

        def convert_function(axis, value):
            if value is not None:
                fun = self._coordinate_to_map_index_single_axis_function(axis, discard_year=True, int_indices=True)
                return fun(value)
            else:
                return value

        ranges = ((t_from, t_to), (x_from, x_to), (y_from, y_to), (z_from, z_to))
        for axis, range_i in enumerate(ranges):
            range_i_from = convert_function(axis, range_i[0])
            range_i_to = convert_function(axis, range_i[1])

            if range_i_from is not None and range_i_to is None:
                set_outside_value_single_axis(axis, left_index=None, right_index=range_i_from)
            elif range_i_from is None and range_i_to is not None:
                set_outside_value_single_axis(axis, left_index=range_i_to, right_index=None)
            elif range_i_from is not None and range_i_to is not None:
                if range_i_from <= range_i_to:
                    set_outside_value_single_axis(axis, left_index=None, right_index=range_i_from)
                    set_outside_value_single_axis(axis, left_index=range_i_to, right_index=None)
                else:
                    set_outside_value_single_axis(axis, left_index=range_i_to, right_index=range_i_from)
        return mask

    def insert_coordinate_values_in_map(self, values, no_data_value=0, land_value=np.nan, skip_values_on_land=True):
        values = np.copy(values)
        values[:, :-1] = self.coordinates_to_map_indices(values[:, :-1], discard_year=True, int_indices=True)
        return self.insert_index_values_in_map(values, no_data_value=no_data_value, land_value=land_value, skip_values_on_land=skip_values_on_land)

    def insert_index_values_in_map(self, values, no_data_value=0, land_value=np.nan, skip_values_on_land=True):
        util.logging.debug('Inserting {} values in map with value {} for no data.'.format(len(values), no_data_value))

        if values.shape[1] not in (4, 5):
            raise ValueError('Values have wrong shape: Second dimension have to be 4 or 5, but it is {}.'.format(values.shape[1]))
        if np.isnan(no_data_value):
            raise ValueError('No data value can not be NAN.')

        # remove time dim if values have no time
        if values.shape[1] == 4:
            old_t_dim = self._t_dim
            self.t_dim = None

        # init map
        dtype = np.promote_types(values.dtype, np.float16)
        value_map = self.masked_map(default_value=no_data_value, land_value=land_value, dtype=dtype)
        number_map = self.masked_map(default_value=0, land_value=0, dtype=np.int32)

        # insert values: sum and count for each box
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

        # average
        mask = number_map > 1
        value_map[mask] = value_map[mask] / number_map[mask]

        # restore time dim
        if values.shape[1] == 4:
            self.t_dim = old_t_dim

        # return
        return value_map

    def value_in_map_from_coordinate(self, coordinate, value_map):
        index = self.coordinate_to_map_index(*coordinate, discard_year=True, int_indices=True)
        return self.value_in_map_from_index(index, value_map)

    def value_in_map_from_index(self, index, value_map):
        if self.dim != value_map.shape:
            raise ValueError(f'Value map must have the same dims as lsm, but its shape is {value_map.shape} and it has to be {self.dim}.')
        return value_map[index]

    # plot
    def plot(self, file=None, use_depth=True, overwrite=False, show_axes=False):
        import util.plot.save
        import matplotlib.pyplot

        scaling_factor = 1 / 10**3

        # get filename
        if file is None:
            filename = str(self)
            if use_depth:
                filename += '_with_depth'
            else:
                filename += '_with_depth_level'
            file = '/tmp/{}.svg'.format(filename)

        # get data
        if use_depth:
            data = self.z[self.lsm] * scaling_factor
        else:
            data = self.lsm

        # plot
        v_max = self.z[-1] * scaling_factor
        util.plot.save.data(file, data, land_value=0, v_max=v_max, overwrite=overwrite, show_axes=show_axes)

    # copy
    def copy(self):
        import copy
        return copy.deepcopy(self)


class LandSeaMaskCache(LandSeaMask):

    def __init__(self, lsm_dir):
        self._lsm_dir = lsm_dir

    @util.cache.file.decorator()
    @overrides.overrides
    def volumes_map(self, t_dim='default', dtype=np.float64):
        return super().volumes_map(t_dim=t_dim, dtype=dtype)

    def volumes_map_cache_file(self, t_dim='default', dtype=np.float64):
        if t_dim == 'default':
            t_dim = self._t_dim
        if t_dim is None or t_dim == 0:
            dtype = np.dtype(dtype)
            filename = measurements.land_sea_mask.constants.VOLUMES_MAP_FILENAME.format(dtype=dtype)
            return os.path.join(self._lsm_dir, filename)
        else:
            return None

    @util.cache.file.decorator()
    @overrides.overrides
    def normalized_volume_weights_map(self, t_dim='default', dtype=np.float64):
        return super().normalized_volume_weights_map(t_dim=t_dim, dtype=dtype)

    def normalized_volume_weights_map_cache_file(self, t_dim='default', dtype=np.float64):
        if t_dim == 'default':
            t_dim = self._t_dim
        if t_dim is None or t_dim == 0:
            dtype = np.dtype(dtype)
            filename = measurements.land_sea_mask.constants.NORMALIZED_VOLUMES_WEIGHTS_MAP_FILENAME.format(dtype=dtype)
            return os.path.join(self._lsm_dir, filename)
        else:
            return None


class LandSeaMaskFromFile(LandSeaMaskCache):

    def __init__(self, lsm_dir, t_dim=None, t_centered=True):
        LandSeaMaskCache.__init__(self, lsm_dir)
        depth = self._calculate_depth()
        lsm = self._calculate_lsm()
        LandSeaMask.__init__(self, lsm, depth, t_dim=t_dim, t_centered=t_centered)

    @property
    def _lsm_file(self):
        return os.path.join(self._lsm_dir, measurements.land_sea_mask.constants.LSM_NPY_FILENAME)

    @property
    def _depth_file(self):
        return os.path.join(self._lsm_dir, measurements.land_sea_mask.constants.DEPTH_NPY_FILENAME)

    @util.cache.file.decorator(cache_file_function=lambda self: self._lsm_file)
    def _calculate_lsm(self):
        raise NotImplementedError

    @util.cache.file.decorator(cache_file_function=lambda self: self._depth_file)
    def _calculate_depth(self):
        raise NotImplementedError


class LandSeaMaskTMM(LandSeaMaskFromFile):

    def __init__(self, t_dim=None, t_centered=True):
        super().__init__(measurements.land_sea_mask.constants.TMM_DIR, t_dim=t_dim, t_centered=t_centered)

    @util.cache.file.decorator(cache_file_function=lambda self: self._lsm_file)
    @overrides.overrides
    def _calculate_lsm(self):
        lsm = util.petsc.universal.load_petsc_mat_to_array(measurements.land_sea_mask.constants.TMM_PETSC_FILE, dtype=np.int16)
        lsm = lsm.transpose()  # metos3d: x and y are changed
        assert lsm.shape == (128, 64) and lsm.min() == 0 and lsm.max() == 15
        return lsm

    @util.cache.file.decorator(cache_file_function=lambda self: self._depth_file)
    @overrides.overrides
    def _calculate_depth(self):
        # read values from txt
        depth = np.genfromtxt(measurements.land_sea_mask.constants.TMM_DEPTH_TXT_FILE, dtype=np.int16, comments='#', usecols=(0,))
        assert depth.ndim == 1 and depth.shape[0] == 16
        return depth

    @property
    def name(self):
        return super().name + '_tmm'


class LandSeaMaskWOA13(LandSeaMaskFromFile):

    def __init__(self, t_dim=None, t_centered=True):
        super().__init__(measurements.land_sea_mask.constants.WOA13_DIR, t_dim=t_dim, t_centered=t_centered)

    @util.cache.file.decorator(cache_file_function=lambda self: self._lsm_file)
    @overrides.overrides
    def _calculate_lsm(self):
        # read values from txt with axis order: x y z
        lsm = np.genfromtxt(measurements.land_sea_mask.constants.WOA13_LSM_TXT_FILE, dtype=float, delimiter=',', comments='#', usecols=(1, 0, 2))

        # normalize values
        lsm[:, 0] = lsm[:, 0] % 360

        lsm = lsm - lsm.min(axis=0)

        # convert to int
        lsm_int = lsm.astype(np.int16)

        assert np.all(lsm_int == lsm)
        assert lsm_int[:, 0].min() == 0 and lsm_int[:, 0].max() == 359 and lsm_int[:, 1].min() == 0 and lsm_int[:, 1].max() == 179 and lsm_int[:, 2].min() == 0 and lsm_int[:, 2].max() == 137

        # convert in 2 dim
        lsm = np.empty((360, 180), dtype=np.int16)
        for x, y, z in lsm_int:
            lsm[x, y] = z

        assert lsm.min() == 0 and lsm.max() == 137
        return lsm

    @util.cache.file.decorator(cache_file_function=lambda self: self._depth_file)
    @overrides.overrides
    def _calculate_depth(self):
        # read values from txt
        depth = np.genfromtxt(measurements.land_sea_mask.constants.WOA13_DEPTH_TXT_FILE, dtype=np.int16, comments='#', usecols=(0,))
        assert depth.ndim == 1 and depth.shape[0] == 138
        return depth

    @property
    def name(self):
        return super().name + '_woa13'


class LandSeaMaskWOA13R(LandSeaMaskCache):

    def __init__(self, t_dim=None, t_centered=True):
        LandSeaMaskCache.__init__(self, measurements.land_sea_mask.constants.WOA13R_DIR)

        depth = measurements.land_sea_mask.depth.values_TMM(max_value=5200, increment_step=2)
        depth.extend([6000, 8000, 10000])

        lsm_woa13 = LandSeaMaskWOA13()
        lsm_woa13.z = depth
        lsm = lsm_woa13.lsm

        LandSeaMask.__init__(self, lsm, depth, t_dim=t_dim, t_centered=t_centered)

    @property
    def name(self):
        return super().name + '_woa13r'
