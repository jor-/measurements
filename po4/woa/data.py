import numpy as np

from util.debug import print_debug

import measurements.po4.woa.regrid_dop
import measurements.po4.woa.regrid_po4
import ndop.metos3d.data


def po4_load_npy_or_save_regrided(npy_file, debug_level = 0, required_debug_level = 1):
    base_string = 'util.measurements.po4.woa.data.po4_load_npy_or_save_regrided: '
    try:
        print_debug(('Loading data from ', npy_file), debug_level, required_debug_level, base_string)
        
        data = np.load(npy_file)
        
    except (OSError, IOError):
        print_debug(('File ', npy_file, ' does not exists. Calculating PO4 data.'), debug_level, required_debug_level, base_string)
        
        measurements.po4.woa.regrid_po4.save_regrided(debug_level=debug_level, required_debug_level=required_debug_level+1)
        
        data = np.load(npy_file)
    
    return data


def po4_nobs(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import PO4_NOBS
    
    data = po4_load_npy_or_save_regrided(PO4_NOBS, debug_level, required_debug_level)
    
    return data



def po4_varis(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import PO4_VARIS
    
    data = po4_load_npy_or_save_regrided(PO4_VARIS, debug_level, required_debug_level)
    
    return data



def po4_means(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import PO4_MEANS
    
    data = po4_load_npy_or_save_regrided(PO4_MEANS, debug_level, required_debug_level)
    
    return data



def dop_load_npy_or_save_regrided(npy_file, debug_level = 0, required_debug_level = 1):
    base_string = 'util.measurements.po4.woa.data.dop_load_npy_or_save_regrided: '
    try:
        print_debug(('Loading data from ', npy_file), debug_level, required_debug_level, base_string)
        
        data = np.load(npy_file)
        
    except (OSError, IOError):
        print_debug(('File ', npy_file, ' does not exists. Calculating DOP data.'), debug_level, required_debug_level, base_string)
        
        land_sea_mask = ndop.metos3d.data.load_land_sea_mask()
        
        measurements.po4.woa.regrid_dop.save_regrided(land_sea_mask, t_dim=12, debug_level=debug_level, required_debug_level=required_debug_level+1)
        
        data = np.load(npy_file)
    
    return data



def dop_nobs(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import DOP_NOBS
    
    data = dop_load_npy_or_save_regrided(DOP_NOBS, debug_level, required_debug_level)
    
    return data


def dop_varis(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import DOP_VARIS
    
    data = dop_load_npy_or_save_regrided(DOP_VARIS, debug_level, required_debug_level)
    
    return data


def dop_means(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import DOP_MEANS
    
    data = dop_load_npy_or_save_regrided(DOP_MEANS, debug_level, required_debug_level)
    
    return data





def npy_or_save_dop_and_po4(npy_file, dop_function, po4_function, debug_level = 0, required_debug_level = 1):
    base_string = 'util.measurements.po4.woa.data.npy_or_save_dop_and_po4: '
    try:
        print_debug(('Loading data from ', npy_file), debug_level, required_debug_level, base_string)
        
        data = np.load(npy_file)
        
    except (OSError, IOError):
        print_debug(('File ', npy_file, ' does not exists. Calculating data.'), debug_level, required_debug_level, base_string)
        
        dop = dop_function(debug_level, required_debug_level)
        po4 = po4_function(debug_level, required_debug_level)
        dop = dop.reshape((1,) + dop.shape)
        po4 = po4.reshape((1,) + po4.shape)
        
        data = np.append(dop, po4, axis=0)
        np.save(npy_file, data)
    
    return data


def nobs(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import NOBS
    
    data = npy_or_save_dop_and_po4(NOBS, dop_nobs, po4_nobs, debug_level, required_debug_level)
    
    return data


def means(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import MEANS
    
    data = npy_or_save_dop_and_po4(MEANS, dop_means, po4_means, debug_level, required_debug_level)
    
    return data


def varis(debug_level = 0, required_debug_level = 1):
    from measurements.po4.woa.constants import VARIS
    
    data = npy_or_save_dop_and_po4(VARIS, dop_varis, po4_varis, debug_level, required_debug_level)
    
    return data
