import numpy as np


def init_masked_map(land_sea_mask=None, default_value=0, dtype=np.float64):
    if land_sea_mask is None:
        from ndop.metos3d.data import load_land_sea_mask
        land_sea_mask = load_land_sea_mask()
    
    z_dim = np.nanmax(land_sea_mask)
    (x_dim, y_dim) = land_sea_mask.shape
    shape = (x_dim, y_dim, z_dim,)
    array = np.ones(shape, dtype=dtype) * default_value
    apply_mask(array, land_sea_mask)
#     array = apply_mask(array, land_sea_mask)

    return array



def apply_mask(array, land_sea_mask=None):
    if land_sea_mask is None:
        from ndop.metos3d.data import load_land_sea_mask
        land_sea_mask = load_land_sea_mask()
    
    (x_dim, y_dim) = land_sea_mask.shape
    for x, y in np.ndindex(x_dim, y_dim):
        z_max = land_sea_mask[x, y]
        array[x, y, z_max:] = np.nan



def insert_values_in_map(values, default_value=0, apply_mask_last=True):
    from ndop.metos3d.constants import METOS_DIM
    
    def insert_space_values_im_map(map, values):
        for row in values:
            (x, y, z, value) = row
            try:
                map[x, y, z] = value
            except IndexError:
                raise ValueError('Space index ' + str((x, y, z)) + ' exceeds dimension ' + str(METOS_DIM) + '.')
        
        if apply_mask_last:
            apply_mask(map)
    
    init_map = init_masked_map(default_value=default_value)
    
    ## if with time axis
    if values.shape[1] == 5:
            time = values[:,0]
            time_unique = np.unique(time)
            time_unique = time_unique[np.argsort(time_unique)]
            time_unique_len = len(time_unique)
            
            value_map = np.empty((time_unique_len,) + METOS_DIM)
            
            for i in range(time_unique_len):
                value_map[i] = np.copy(init_map)
                values_i = values[time == time_unique[i], 1:]
                insert_space_values_im_map(value_map[i], values_i)
    ## if without time axis
    elif values.shape[1] == 4:
        insert_space_values_im_map(value_map, values)
        value_map = init_map
    else:
        raise Exception('Values have wrong shape: Second dimension have to be 4 or 5, but it is ' + str(values.shape[1]) + '.')
    
    return value_map
    


def average_over_time(data):
    averaged_data = np.nansum(data, axis=0) / np.sum(np.logical_not(np.isnan(data)), axis=0)
    return averaged_data



def wrap_around_index(index, index_range):
    if index_range is not None:
        index_range_diff = index_range[1] - index_range[0]
        while index < index_range[0]:
            index += index_range_diff
        while index >= index_range[1]:
            index -= index_range_diff
    return index



def get_min_distance(point_1, point_2, t_range=None, x_range=None):
    distance = np.abs(point_1 - point_2)
    
    ## wrap around
    if t_range is not None:
        t_size = t_range[1] - t_range[0]
        if distance[0] > t_size / 2:
            distance[0] -= t_size / 2
            
    if x_range is not None:
        x_size = x_range[1] - x_range[0]
        if distance[1] > x_size / 2:
            distance[1] -= x_size / 2
    
    return distance