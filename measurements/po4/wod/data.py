import overrides

import util.cache.file_based
import util.cache.memory_based

import measurements.universal.data
import measurements.universal.constants
import measurements.constants
import measurements.po4.wod.values
import measurements.po4.constants
import measurements.po4.wod.constants



class Measurements(measurements.universal.data.MeasurementsAnnualPeriodicCache):
    
    def __init__(self, sample_t_dim=measurements.po4.wod.constants.SAMPLE_T_DIM, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        
        tracer = 'po4'        
        data_set_name = 'wod_2013'        
        
        sample_lsm = measurements.po4.wod.constants.SAMPLE_LSM
        sample_lsm.t_dim = sample_t_dim
        min_deviation = measurements.po4.constants.DEVIATION_MIN_VALUE
        
        super().__init__(sample_lsm, tracer=tracer, data_set_name=data_set_name, min_standard_deviation=min_deviation, min_measurements_correlations=min_measurements_correlations)
        
        self.fill_strategy = 'interpolate'
        
        try:
            INTERPOLATOR_OPTIONS = measurements.po4.wod.constants.INTERPOLATOR_OPTIONS
        except AttributeError:
            pass
        else:
            try:
                interpolator_option = INTERPOLATOR_OPTIONS['mean']['concentration'][measurements.constants.MEAN_MIN_MEASUREMENTS][str(sample_lsm)]
            except KeyError:
                pass
            else:
                self.set_interpolator_options('concentration_means', interpolator_option)
            try:
                interpolator_option = INTERPOLATOR_OPTIONS['deviation']['concentration'][measurements.constants.DEVIATION_MIN_MEASUREMENTS][str(sample_lsm)]
            except KeyError:
                pass
            else:
                self.set_interpolator_options('concentration_standard_deviations', interpolator_option)
            try:
                interpolator_option = INTERPOLATOR_OPTIONS['deviation']['average_noise'][measurements.constants.DEVIATION_MIN_MEASUREMENTS][str(sample_lsm)]
            except KeyError:
                pass
            else:
                self.set_interpolator_options('average_noise_standard_deviations', interpolator_option)

    
    def __str__(self):
        string = super().__str__()
        if self.min_measurements_correlations < float('inf'):
            string = string + '({min_measurements_correlations})'.format(min_measurements_correlations=self.min_measurements_correlations)
        return string


    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    @overrides.overrides
    def points(self):
        return measurements.po4.wod.values.points()

    @property
    @util.cache.memory_based.decorator()
    @util.cache.file_based.decorator()
    @overrides.overrides
    def values(self):
        return measurements.po4.wod.values.results()



class MeasurementsNearWater(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):
    
    def __init__(self, water_lsm=None, max_box_distance_to_water=0, sample_t_dim=measurements.po4.wod.constants.SAMPLE_T_DIM, min_measurements_correlations=measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS):
        measurements = Measurements(sample_t_dim=sample_t_dim, min_measurements_correlations=min_measurements_correlations)
        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)

