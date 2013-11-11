from .model import Covariance_Model

import numpy as np
from petsc4py import PETSc as petsc

import logging
logger = logging.getLogger(__name__)


def create_covariance_matrix(n):
    covariance_model = Covariance_Model()
    
    A = np.empty((n, n))
    
    for i, j in np.ndindex(n, n):
        A[i, j] = covariance_model.covariance(i, j)
    
    return A
    


class Covariance_Matrix_Shell_Petsc:
    
    def __init__(self):
        self.covariance_model = Covariance_Model()
    
    @property
    def n(self):
        return self.covariance_model.n
    
    def mult(self, context, x, y):
        logger.debug('Multiplying covariance matrix with vector without explicit matrix.')
        
        ## copy x to local vec
        scatter, x_local = petsc.Scatter.toAll(x)
        scatter.scatterBegin(x, x_local)
        scatter.scatterEnd(x, x_local)
        scatter.destroy()
        
        
        ## set y values
        y_ownership_range = y.getOwnershipRange()
        y_size_local = y_ownership_range[1] - y_ownership_range[0]
        y_size_global = y.getSize()
        
        for i_local in range(y_size_local):
            i_global = y_ownership_range[0] + i_local
            
            ## compute value
            value = 0
            for j_global in range(y_size_global):
                value += self.covariance_model.covariance(i_global, j_global) * x_local.getValue(j_global)
            
            y.setValue(i_global, value)
        y.assemblyBegin()
        y.assemblyEnd()
        
        ## destroy local copy
        x_local.destroy()


def create_covariance_matrix_petsc(n=None):
    shell = Covariance_Matrix_Shell_Petsc()
    if n is None:
        n = shell.n
    
    logger.debug('Creating covariance matrix in petsc format with size %d.' % n)
    
    A = petsc.Mat()
    A.createPython([n,n], context=shell, comm=petsc.COMM_WORLD)
    
    return A