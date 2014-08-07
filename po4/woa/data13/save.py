import measurements.po4.woa.data13.regrid
import measurements.land_sea_mask.load
import util.io

def measurements_to_metos_boxes():
    from measurements.land_sea_mask.constants import LSM_128x64x15_Z_LEFT as z_values_left
    from .constants import MEANS_FILE, NOBS_FILE, VARIANCES_FILE
    
    land_sea_mask = measurements.land_sea_mask.load.resolution_128x64x15()
    
    (means, nobs, variances) = measurements.po4.woa.data13.regrid.measurements_to_land_sea_mask(land_sea_mask, z_values_left, t_dim=12)
    
    for (value, file) in [[means, MEANS_FILE], [nobs, NOBS_FILE], [variances, VARIANCES_FILE]]:
        util.io.save_npy(value, file, make_read_only=True, create_path_if_not_exists=True)
    