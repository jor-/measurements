import numpy as np


def init_masked_map(land_sea_mask=None, no_data_value=0, dtype=np.float64):
    if land_sea_mask is None:
        from ndop.metos3d.data import load_land_sea_mask
        land_sea_mask = load_land_sea_mask()
    
    z_dim = np.nanmax(land_sea_mask)
    (x_dim, y_dim) = land_sea_mask.shape
    shape = (x_dim, y_dim, z_dim,)
    array = np.ones(shape, dtype=dtype) * no_data_value
    apply_mask(array, land_sea_mask)

    return array



def apply_mask(array, land_sea_mask=None):
    if land_sea_mask is None:
        from ndop.metos3d.data import load_land_sea_mask
        land_sea_mask = load_land_sea_mask()
    
    (x_dim, y_dim) = land_sea_mask.shape
    for x, y in np.ndindex(x_dim, y_dim):
        z_max = land_sea_mask[x, y]
        array[x, y, z_max:] = np.nan



def insert_values_in_map(values, no_data_value=0, apply_mask_last=True):
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
    
    init_map = init_masked_map(no_data_value=no_data_value)
    
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
        value_map = init_map
        insert_space_values_im_map(value_map, values)
    else:
        raise Exception('Values have wrong shape: Second dimension have to be 4 or 5, but it is ' + str(values.shape[1]) + '.')
    
    return value_map
    