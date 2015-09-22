import numpy as np
import scipy.sparse

import measurements.all.pw.correlation
import measurements.all.pw_nearest.data
import measurements.all.pw_nearest.constants
import measurements.land_sea_mask.data

import util.cache
import util.logging
logger = util.logging.logger


class CorrelationMatrix(measurements.all.pw.correlation.CorrelationMatrix):
    
    def __init__(self, *args, lsm=measurements.land_sea_mask.data.LandSeaMaskTMM(), max_land_boxes=0, **kargs):
        logger.debug('Initiating {} with max_land_boxes {}.'.format(lsm, max_land_boxes))
        
        self.lsm = lsm
        self.max_land_boxes = max_land_boxes

        all_correlation_matix_object = measurements.all.pw.correlation.CorrelationMatrix(*args, **kargs)
        all_correlation_matix_object.memory_cache_switch(False)
        self.all_correlation_matix_object = all_correlation_matix_object
        
        super().__init__(*args, **kargs)
        CORRELATION_DIR = measurements.all.pw_nearest.constants.CORRELATION_DIR.format(lsm=lsm, max_land_boxes=max_land_boxes)
        self.object_cache.cache_dir = CORRELATION_DIR
        self.npy_cache.cache_dir = CORRELATION_DIR
    
    
    ## nearest
    
    def near_water_matrix_calculate(self):
        mask = measurements.all.pw_nearest.data.points_near_water_mask_concatenated(lsm=self.lsm, max_land_boxes=self.max_land_boxes)
        n = len(mask)
        m = mask.sum()
        near_water_matrix = scipy.sparse.dok_matrix((n, m), dtype=np.int16)
        
        j = 0
        for i in range(n):
            if mask[i]:
                near_water_matrix[i, j] = 1
                j = j + 1
        assert j == m
        
        return near_water_matrix
        
    
    def near_water_matrix(self):
        return self.memory_cache[('near_water_matrix', self.near_water_matrix_calculate)]
    
    
    def project_matrix(self, A, format='csc', dtype=np.float32):
        D = self.near_water_matrix()
        A = D.T * A * D
        return A.asformat(format).astype(dtype)
    

    ## sample matrix

    def sample_quantity_matrix_calculate(self, format='csc', dtype=np.float32):
        return self.project_matrix(self.all_correlation_matix_object.sample_quantity_matrix(), format=format, dtype=dtype)

    ## correlation matrix

    def correlation_matrix_calculate(self, format='csc', dtype=np.float32):
        return self.project_matrix(self.all_correlation_matix_object.correlation_matrix(), format=format, dtype=dtype)

