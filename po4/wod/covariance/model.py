import measurements.po4.wod.values as values
from ..correlation.model import Correlation_Model

import logging


class Covariance_Model():
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.standard_deviations = values.load_standard_deviations()
        
        correlation_model = Correlation_Model()
        self.correlation_function = lambda i, j: correlation_model.correlation(i, j)
    
    @property
    def n(self):
        return len(self.standard_deviations)
    
    
    def covariance(self, i, j):
        correlation = self.correlation_function(i, j)
        covariance = correlation * self.standard_deviations[i] * self.standard_deviations[j]
        
        return covariance