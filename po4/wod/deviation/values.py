import measurements.po4.wod.data.interpolate

import measurements.po4.constants as PO4_CONSTANTS
import measurements.po4.wod.deviation.constants as CONSTANTS


class Interpolator(measurements.po4.wod.data.interpolate.Interpolator):
    def __init__(self):
        data_function = lambda m: m.deviations(minimum_measurements=PO4_CONSTANTS.DEVIATION_MIN_MEASUREMENTS, min_deviation=PO4_CONSTANTS.DEVIATION_MIN_VALUE)
        super().__init__(data_function, CONSTANTS.DEVIATION_DIR, CONSTANTS.INTERPOLATED_DEVIATION_FILENAME)
    
    
    def data_for_points(self):
        return super().data_for_points(interpolator_setup=(0.1, 2, 0.2, 1))
    
    def data_for_TMM(self, t_dim):
        return super().data_for_TMM(t_dim=t_dim, interpolator_setup=(0.1, 2, 0.2, 0))
    
    def data_for_WOA13(self, t_dim):
        return super().data_for_WOA13(t_dim=t_dim, interpolator_setup=(0.1, 2, 0.2, 0))
    
    def data_for_WOA13R(self, t_dim):
        return super().data_for_WOA13R(t_dim=t_dim, interpolator_setup=(0.1, 2, 0.2, 0))



def for_points():
    return Interpolator().data_for_points()

def for_TMM(t_dim=48):
    return Interpolator().data_for_TMM(t_dim=t_dim)
    
def for_WOA13(t_dim=48):
    return Interpolator().data_for_WOA13(t_dim=t_dim)
    
def for_WOA13R(t_dim=48):
    return Interpolator().data_for_WOA13R(t_dim=t_dim)
    
