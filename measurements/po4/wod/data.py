import util.cache.file_based
import util.cache.memory_based

import measurements.universal.data
import measurements.universal.constants
import measurements.constants
import measurements.po4.wod.values
import measurements.po4.constants
import measurements.po4.wod.constants



class Measurements(measurements.universal.data.MeasurementsAnnualPeriodicFillInterpolationCache):
    
    def __init__(self, sample_t_dim=measurements.po4.wod.constants.SAMPLE_T_DIM, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        
        tracer = 'po4'        
        data_set_name = 'wod_2013'        
        
        sample_lsm = measurements.po4.wod.constants.SAMPLE_LSM
        sample_lsm.t_dim = sample_t_dim
        min_deviation = measurements.po4.constants.DEVIATION_MIN_VALUE
        
        mean_interpolator_setup = measurements.po4.wod.constants.INTERPOLATOR_SETUPS['mean']['concentration'][measurements.constants.MEAN_MIN_MEASUREMENTS][str(sample_lsm)]
        concentation_deviation_interpolator_setup = measurements.po4.wod.constants.INTERPOLATOR_SETUPS['deviation']['concentration'][measurements.constants.DEVIATION_MIN_MEASUREMENTS][str(sample_lsm)]
        average_noise_deviation_interpolator_setup = measurements.po4.wod.constants.INTERPOLATOR_SETUPS['deviation']['average_noise'][measurements.constants.DEVIATION_MIN_MEASUREMENTS][str(sample_lsm)]
        
        super().__init__(mean_interpolator_setup, concentation_deviation_interpolator_setup, average_noise_deviation_interpolator_setup, sample_lsm, tracer=tracer, data_set_name=data_set_name, min_deviation=min_deviation, min_measurements_correlations=min_measurements_correlations)

    
    def __str__(self):
        string = super().__str__()
        if self.min_measurements_correlations < float('inf'):
            string = string + '({min_measurements_correlations})'.format(min_measurements_correlations=self.min_measurements_correlations)
        return string


    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def points(self):
        return measurements.po4.wod.values.points()

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    def values(self):
        return measurements.po4.wod.values.results()



class MeasurementsNearWater(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):
    
    def __init__(self, water_lsm=None, max_box_distance_to_water=0, sample_t_dim=measurements.po4.wod.constants.SAMPLE_T_DIM, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurements = Measurements(sample_t_dim=sample_t_dim, min_measurements_correlations=min_measurements_correlations)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)

