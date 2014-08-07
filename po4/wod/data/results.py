import measurements.util.data

from .constants import MEASUREMENTS_DICT_UNSORTED_FILE, MEASUREMENTS_DICT_SORTED_FILE


# def add_cruises_to_measurements(measurements, cruises):
#     measurements_dict = measurements.measurements_dict
#     
#     ## insert results in dict
#     for cruise in cruises:
#         x = cruise.x
#         y = cruise.y
#         z = cruise.z
#         t = cruise.dt_float
#         results = cruise.po4.astype(float)
#         
#         for i in range(results.size):
#             index = (t, x, y, z[i])
#             measurements.add_result(index, results[i])


class Measurements_Unsorted(measurements.util.data.Measurements_Unsorted):
    
    def __init__(self):
        super().__init__()
    
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
    
    
    def save(self, file=MEASUREMENTS_DICT_UNSORTED_FILE):
        super().save(file)
    
    def load(self, file=MEASUREMENTS_DICT_UNSORTED_FILE):
        super().load(file)


class Measurements_Sorted(measurements.util.data.Measurements_Sorted, Measurements_Unsorted):
    
    def __init__(self):
        super().__init__()
    
    
    def save(self, file=MEASUREMENTS_DICT_SORTED_FILE):
        super().save(file)
    
    def load(self, file=MEASUREMENTS_DICT_SORTED_FILE):
        super().load(file)

