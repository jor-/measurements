from .model import Correlation_Model

import numpy as np
from petsc4py import PETSc as petsc

from util.petsc.with_petsc4py import Matrix_Shell_Petsc

import logging
logger = logging.getLogger(__name__)


def create_correlation_matrix(n):
    model = Correlation_Model()
    
    A = np.empty((n, n))
    
    for i, j in np.ndindex(n, n):
        A[i, j] = model.correlaton(i, j)
    
    return A
    


# class Correlation_Matrix_Shell_Petsc:
#     
#     def __init__(self):
#         self.correlation_model = Correlation_Model()
#     
#     @property
#     def n(self):
#         return self.covariance_model.n
#     
#     def mult(self, context, x, y):
#         logger.debug('Multiplying correlation matrix with vector without explicit matrix.')
#         
#         ## copy x to local vec
#         scatter, x_local = petsc.Scatter.toAll(x)
#         scatter.scatterBegin(x, x_local)
#         scatter.scatterEnd(x, x_local)
#         scatter.destroy()
#         
#         
#         ## set y values
#         y_ownership_range = y.getOwnershipRange()
#         y_size_local = y_ownership_range[1] - y_ownership_range[0]
#         y_size_global = y.getSize()
#         
#         for i_local in range(y_size_local):
#             i_global = y_ownership_range[0] + i_local
#             
#             ## compute value
#             value = 0
#             for j_global in range(y_size_global):
#                 value += self.correlation_model.correlation(i_global, j_global) * x_local.getValue(j_global)
#             
#             y.setValue(i_global, value)
#         y.assemblyBegin()
#         y.assemblyEnd()
#         
#         ## destroy local copy
#         x_local.destroy()


def create_covariance_matrix_petsc(n=None):
    model = Correlation_Model()
    entry_function = lambda i, j: model.correlation(i, j)
    shell = Matrix_Shell_Petsc(entry_function)
    if n is None:
        n = shell.n
    
    logger.debug('Creating covariance matrix in petsc format with size %d.' % n)
    
    A = petsc.Mat()
    A.createPython([n,n], context=shell, comm=petsc.COMM_WORLD)
    
    return A