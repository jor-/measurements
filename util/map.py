import numpy as np


def init_masked_map(land_sea_mask=None, default_value=0, dtype=np.float64, t_dim=None):
    if land_sea_mask is None:
        from ndop.model.data import load_land_sea_mask
        land_sea_mask = load_land_sea_mask()
    
    z_dim = np.nanmax(land_sea_mask)
    (x_dim, y_dim) = land_sea_mask.shape
    shape = (x_dim, y_dim, z_dim)
    masked_map = np.ones(shape, dtype=dtype) * default_value
    apply_mask(masked_map, land_sea_mask)
    
    if t_dim is not None:
        masked_map_with_t = np.empty((t_dim,) +  masked_map.shape)
        for i in range(t_dim):
            masked_map_with_t[i] = masked_map
        return masked_map_with_t
    else:
        return masked_map

# def init_masked_array(land_sea_mask, t_dim, dtype=np.float64):
#     z_dim = np.nanmax(land_sea_mask)
#     (x_dim, y_dim) = land_sea_mask.shape
#     shape = (t_dim, x_dim, y_dim, z_dim,)
#     array = np.zeros(shape, dtype=dtype)
#     
#     for x, y in np.ndindex(x_dim, y_dim):
#         z_max = land_sea_mask[x, y]
#         array[:, x, y, z_max:] = np.nan
# 
#     return array



def apply_mask(array, land_sea_mask=None, land_value=np.nan):
    import ndop.model.data
    if land_sea_mask is None:
        land_sea_mask = ndop.model.data.load_land_sea_mask()
    
    (x_dim, y_dim) = land_sea_mask.shape
    for x, y in np.ndindex(x_dim, y_dim):
        z_max = land_sea_mask[x, y]
        array[x, y, z_max:] = land_value
    
    return array



def insert_values_in_map(values, no_data_value=0, apply_mask_last=True):
    from ndop.model.constants import METOS_DIM
    import ndop.model.data
    land_sea_mask = ndop.model.data.load_land_sea_mask()
    
    def insert_space_values_im_map(map, values):
        for row in values:
            (x, y, z, value) = row
            try:
                map[x, y, z] = value
            except IndexError:
                raise ValueError('Space index ' + str((x, y, z)) + ' exceeds dimension ' + str(METOS_DIM) + '.')
        
        if apply_mask_last:
            apply_mask(map, land_sea_mask=land_sea_mask)
    
    init_map = init_masked_map(land_sea_mask=land_sea_mask, default_value=no_data_value)
    
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
    