import os.path
import numpy as np

import measurements.po4.metos_boxes.mean.io
import measurements.util.map

import ndop.model.data

import util.petsc.universal
import util.logging
# logger = util.logging.get_logger()

TRAJECTORY_FILENAME = 'ts{:0>4}-PO4-jor.petsc'
LOG_FILENAME = 'log.txt'

def save(path='/work_O2/sunip229/transit/Jarek/trajectories/PO4'):
    METOS_T_DIM = 2880
    t_dim = 2880
    interpolator_setup=(0.1,1,0.0)
    
    log_file = os.path.join(path, LOG_FILENAME)
    
    with util.logging.Logger(log_file=log_file) as logger:
        logger = logger.logger
        
        interpolated_map = measurements.po4.metos_boxes.mean.io.load_interpolated_map(t_dim, interpolator_setup=interpolator_setup)
        
        file = os.path.join(path, TRAJECTORY_FILENAME)
        
        do_interpolation = t_dim != METOS_T_DIM
        if do_interpolation:
            ## init t indices and values for interpolation
            data_t_indices = np.arange(-1, t_dim+1)
            data_t_values = (data_t_indices + 0.5) * METOS_T_DIM / t_dim
            
            def get_interpolated_map_at_t_indices_index(t_indices_index):
                t_index = data_t_indices[t_indices_index]
                
                if t_index < 0:
                    t_index = 0
                if t_index >= len(interpolated_map):
                    t_index = len(interpolated_map) - 1
                
                return interpolated_map[t_index]
            
            data_t_indices_index = -1
            
        
        for trajector_t_index in range(METOS_T_DIM):
            if do_interpolation:
                ## get (left) data t index
                while data_t_values[data_t_indices_index + 1] < trajector_t_index:
                    data_t_indices_index += 1
                
                ## interpolate map linear for t index
                interpolation_fraction = (trajector_t_index - data_t_values[data_t_indices_index]) / (data_t_values[data_t_indices_index + 1] - data_t_values[data_t_indices_index])
                interpolated_map_at_trajector_t_index = (1 - interpolation_fraction) * get_interpolated_map_at_t_indices_index(data_t_indices_index) + interpolation_fraction * get_interpolated_map_at_t_indices_index(data_t_indices_index + 1)
                
                logger.debug('Interpolating values for t trajecory index {} with t data index {} and index interpolation fraction {}'.format(trajector_t_index, data_t_indices_index, interpolation_fraction))
            else:
                ## get map for t index
                interpolated_map_at_trajector_t_index = interpolated_map[trajector_t_index]
            
            ## convert to metos vector and save
            petsc_vec_t_index = ndop.model.data.convert_3D_to_metos_1D(interpolated_map_at_trajector_t_index)
            util.petsc.universal.save_numpy_array_to_petsc_vec(petsc_vec_t_index , file.format(trajector_t_index))
            


save()