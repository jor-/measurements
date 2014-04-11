import numpy as np
import logging
logger = logging.getLogger(__name__)

import ndop.model.data
import measurements.dop.pw.data
import measurements.util.map
import util.io



def save(land_sea_mask, t_dim=12):
    from measurements.dop.constants import DEVIATION_MIN_MEASUREMENTS, DEVIATION_MIN_VALUE
    from .constants import NOBS_FILE, VARIS_FILE, MEANS_FILE
    
    logger.debug('Calculating and saving dop measurement data.')
    
    ## load measurements
    measurement_data = measurements.dop.pw.data.load_data()
    
    ## init values
    nobs = measurements.util.map.init_masked_map(land_sea_mask=land_sea_mask, t_dim=t_dim, default_value=0, dtype=np.float64)
    sum_of_values = np.copy(nobs)
    sum_of_squares = np.copy(nobs)
    varis = measurements.util.map.init_masked_map(land_sea_mask=land_sea_mask, t_dim=t_dim, default_value=np.inf, dtype=np.float64)
    
    number_of_measurements = measurement_data.shape[0]
    
    ## insert measurements
    for i in range(number_of_measurements):
        t, x, y, z, dop = measurement_data[i, :]
        (t_index, x_index, y_index, z_index) = ndop.model.data.convert_point_to_metos_index(t, x, y, z, t_dim, land_sea_mask)
        
        assert nobs[t_index, x_index, y_index, z_index] is not np.nan
        assert sum_of_values[t_index, x_index, y_index, z_index] is not np.nan
        assert sum_of_squares[t_index, x_index, y_index, z_index] is not np.nan
        
        nobs[t_index, x_index, y_index, z_index] += 1
        sum_of_values[t_index, x_index, y_index, z_index] += dop
        sum_of_squares[t_index, x_index, y_index, z_index] += dop**2
    
    ## average measurements
    where_measurements = np.where(nobs > 0)
    
    mean = np.copy(sum_of_values)
    mean[where_measurements] /= nobs[where_measurements]
    
    ## calculate averaged variance and assign
    where_measurements_over_threshold = np.where(nobs >= DEVIATION_MIN_MEASUREMENTS)
    nobs_over_threshold = nobs[where_measurements_over_threshold]
    sum_of_values_over_threshold = sum_of_values[where_measurements_over_threshold]
    sum_of_squares_over_threshold = sum_of_squares[where_measurements_over_threshold]
    
    varis[where_measurements_over_threshold] = (sum_of_squares_over_threshold - sum_of_values_over_threshold ** 2 / nobs_over_threshold) / (nobs_over_threshold - 1)
    varis[np.where(varis < DEVIATION_MIN_VALUE**2)] = DEVIATION_MIN_VALUE**2
#     vari_averaged = np.nansum(varis) / (varis > 0).sum()
#     varis[np.logical_not(np.isnan(varis))] = vari_averaged
    vari_averaged = varis[np.isfinite(varis)].mean()
    varis[varis == np.inf] = vari_averaged
    
    ## save values
    util.io.save_npy(NOBS_FILE, nobs, make_read_only=True, create_path_if_not_exists=True)
    util.io.save_npy(VARIS_FILE, varis, make_read_only=True, create_path_if_not_exists=True)
    util.io.save_npy(MEANS_FILE, mean, make_read_only=True, create_path_if_not_exists=True)