import os.path

import measurements.util.data

from .constants import DATA_DIR, MEASUREMENTS_DICT_UNSORTED_FILENAME, MEASUREMENTS_DICT_SORTED_FILENAME



class Measurements(measurements.util.data.Measurements):
    
    def __init__(self, sorted=False):
        super().__init__(sorted=sorted)
    
    def add_cruises(self, cruises):
        measurements_dict = self.measurements_dict
        
        ## insert results in dict
        for cruise in cruises:
            x = cruise.x
            y = cruise.y
            z = cruise.z
            t = cruise.dt_float
            results = cruise.po4.astype(float)
            
            for i in range(results.size):
                index = (t, x, y, z[i])
                self.add_result(index, results[i])    
    
    
    def save(self, file=os.path.join(DATA_DIR, MEASUREMENTS_DICT_UNSORTED_FILENAME)):
        super().save(file)
    
    
    @classmethod
    def load(cls, file=os.path.join(DATA_DIR, MEASUREMENTS_DICT_UNSORTED_FILENAME)):
        return super().load(file)



class MeasurementsUnsorted(Measurements):
    
    def __init__(self):
        super().__init__(sorted=False)



class MeasurementsSorted(Measurements):
    
    def __init__(self):
        super().__init__(sorted=False)
    
    def save(self, file=os.path.join(DATA_DIR, MEASUREMENTS_DICT_SORTED_FILENAME)):
        super().save(file)
    
    @classmethod
    def load(cls, file=os.path.join(DATA_DIR, MEASUREMENTS_DICT_SORTED_FILENAME)):
        return super().load(file)

