import numpy as np

import measurements.po4.wod.data.interpolate

import measurements.po4.wod.deviation.constants as CONSTANTS


DEFAULT_INTERPOLATOR_SETUPS = {
'average_noise': {3: {'lsm_12_woa13r':(0.1,6,0.6,1), 'lsm_12_tmm':(0.1,6,0.6,1)}, 4: {'lsm_12_woa13r':(0.2,2,0.4,1)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}},
'concentration': {3: {'lsm_12_woa13r':(0.2,3,0.6,1), 'lsm_12_tmm':(0.2,1,0.0,0)}, 4: {'lsm_12_woa13r':(0.2,1,0,0)}, 5: {'lsm_12_woa13r':(0.2,1,0,0)}}
}


class InterpolatorDirect(measurements.po4.wod.data.interpolate.Interpolator):

    def __init__(self, min_values=CONSTANTS.MIN_MEASUREMENTS, sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=None):
        self.min_values = min_values
        filename = CONSTANTS.INTERPOLATED_DEVIATION_DIRECT_FILENAME.format(min_values=min_values, sample_lsm=sample_lsm, scaling_values='{scaling_values}', points='{points}', interpolator_setup='{interpolator_setup}')
        super().__init__(CONSTANTS.DEVIATION_DIR, filename, sample_lsm=sample_lsm, scaling_values=scaling_values)


    @property
    def data(self):
        m = measurements.po4.wod.data.values.measurement_dict()
        m.categorize_indices_to_lsm(self.sample_lsm, discard_year=False)
        m.discard_year()
        m.deviations(min_values=self.min_values, min_deviation=CONSTANTS.MIN_MEASUREMENTS, return_type='self')
        return m.items()




class InterpolatorConcentration(measurements.po4.wod.data.interpolate.Interpolator):

    def __init__(self, min_values=CONSTANTS.MIN_MEASUREMENTS, sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=None):
        self.min_values = min_values
        filename = CONSTANTS.INTERPOLATED_DEVIATION_CONCENTRATION_FILENAME.format(min_values=min_values, sample_lsm=sample_lsm, scaling_values='{scaling_values}', points='{points}', interpolator_setup='{interpolator_setup}')
        try:
            default_interpolator_setups = DEFAULT_INTERPOLATOR_SETUPS['concentration'][min_values]
        except KeyError:
            default_interpolator_setups = None
        super().__init__(CONSTANTS.DEVIATION_DIR, filename, sample_lsm=sample_lsm, scaling_values=scaling_values, default_interpolator_setups=default_interpolator_setups)


    @property
    def data(self):
        m = measurements.po4.wod.data.values.measurement_dict()
        m.categorize_indices_to_lsm(self.sample_lsm, discard_year=False)
        m.means(min_values=1, return_type='self')
        m.discard_year()
        m.deviations(min_values=self.min_values, min_deviation=0, return_type='self')
        return m.items()




class InterpolatorAverageNoise(measurements.po4.wod.data.interpolate.Interpolator):

    def __init__(self, min_values=CONSTANTS.MIN_MEASUREMENTS, sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=None):
        self.min_values = min_values
        filename = CONSTANTS.INTERPOLATED_DEVIATION_AVERAGE_ERROR_FILENAME.format(min_values=min_values, sample_lsm=sample_lsm, scaling_values='{scaling_values}', points='{points}', interpolator_setup='{interpolator_setup}')
        try:
            default_interpolator_setups = DEFAULT_INTERPOLATOR_SETUPS['average_noise'][min_values]
        except KeyError:
            default_interpolator_setups = None
        super().__init__(CONSTANTS.DEVIATION_DIR, filename, sample_lsm=sample_lsm, scaling_values=scaling_values, default_interpolator_setups=default_interpolator_setups)


    @property
    def data(self):
        m = measurements.po4.wod.data.values.measurement_dict()
        m.categorize_indices_to_lsm(self.sample_lsm, discard_year=False)
        m.deviations(min_values=self.min_values, min_deviation=CONSTANTS.MIN_VALUE, return_type='self')
        m.discard_year()
        m.means(min_values=1, return_type='self')
        return m.items()




class InterpolatorTotal(measurements.po4.wod.data.interpolate.Interpolator):

    def __init__(self, min_values=CONSTANTS.MIN_MEASUREMENTS, sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=None):
        self.min_values = min_values
        filename = CONSTANTS.INTERPOLATED_DEVIATION_TOTAL_FILENAME.format(min_values=min_values, sample_lsm=sample_lsm, scaling_values='{scaling_values}', points='{points}', concentration_interpolator_setup='{concentration_interpolator_setup}', average_error_interpolator_setup='{average_error_interpolator_setup}')
        
        super().__init__(CONSTANTS.DEVIATION_DIR, filename, sample_lsm=sample_lsm, scaling_values=scaling_values)

        self.interpolator_concentration = InterpolatorConcentration(min_values=min_values, sample_lsm=sample_lsm, scaling_values=scaling_values)
        self.interpolator_average_noise = InterpolatorAverageNoise(min_values=min_values, sample_lsm=sample_lsm, scaling_values=scaling_values)


    def noise(self, average_error_interpolator_setup=None):
        noise_dict = measurements.po4.wod.data.values.measurement_dict()
        noise_dict.categorize_indices_to_lsm(self.sample_lsm, discard_year=False)
        noise_dict.deviations(min_values=self.min_values, min_deviation=CONSTANTS.MIN_MEASUREMENTS, return_type='self')
        
        points = measurements.po4.wod.data.values.points()
        average_noise = self.interpolator_average_noise.interpolated_data_for_points(interpolator_setup=average_error_interpolator_setup)
        assert len(points) == len(average_noise)
        
        n = len(points)
        noise = np.empty(n)
        for i in range(n):
            index_i = noise_dict.categorize_index_to_lsm(points[i], self.sample_lsm, discard_year=False)
            noise_list_i = noise_dict[index_i]
            assert len(noise_list_i) in [0, 1]
            
            if len(noise_list_i) == 1:
                noise[i] = noise_list_i[0]
            else:
                noise[i] = average_noise[i]
                
        assert np.all(np.isfinite(noise))                
        assert np.all(noise > 0)
        return noise


    def interpolated_data_for_points_calculate(self, concentration_interpolator_setup, average_error_interpolator_setup):
        concentration = self.interpolator_concentration.interpolated_data_for_points(interpolator_setup=concentration_interpolator_setup)
        noise = self.noise(average_error_interpolator_setup=average_error_interpolator_setup)
        return (concentration**2 + noise**2)**(1/2)


    def interpolated_data_for_points(self, concentration_interpolator_setup=None, average_error_interpolator_setup=None):
        concentration_interpolator_setup = self.interpolator_concentration.get_interpolator_setup(concentration_interpolator_setup)
        average_error_interpolator_setup = self.interpolator_average_noise.get_interpolator_setup(average_error_interpolator_setup)
        
        filename = self.interpolated_data_filename.format(points='lexsorted_points', concentration_interpolator_setup=self._tuple_to_str(concentration_interpolator_setup), average_error_interpolator_setup=self._tuple_to_str(average_error_interpolator_setup), scaling_values=self._tuple_to_str(self.scaling_values))
        function = lambda :self.interpolated_data_for_points_calculate(concentration_interpolator_setup=concentration_interpolator_setup, average_error_interpolator_setup=average_error_interpolator_setup)
        return self.cache.get_value(filename, function)
        

    def interpolated_data_for_lsm_calculate(self, lsm, concentration_interpolator_setup, average_error_interpolator_setup):
        concentration = self.interpolator_concentration.interpolated_data_for_lsm(lsm, interpolator_setup=concentration_interpolator_setup)
        average_noise = self.interpolator_average_noise.interpolated_data_for_lsm(lsm, interpolator_setup=average_error_interpolator_setup)
        return (concentration**2 + average_noise**2)**(1/2)


    def interpolated_data_for_lsm(self, lsm, concentration_interpolator_setup=None, average_error_interpolator_setup=None):
        concentration_interpolator_setup = self.interpolator_concentration.get_interpolator_setup(concentration_interpolator_setup)
        average_error_interpolator_setup = self.interpolator_average_noise.get_interpolator_setup(average_error_interpolator_setup)
        
        filename = self.interpolated_data_filename.format(points=lsm, concentration_interpolator_setup=self._tuple_to_str(concentration_interpolator_setup), average_error_interpolator_setup=self._tuple_to_str(average_error_interpolator_setup), scaling_values=self._tuple_to_str(self.scaling_values))
        function = lambda :self.interpolated_data_for_lsm_calculate(lsm, concentration_interpolator_setup=concentration_interpolator_setup, average_error_interpolator_setup=average_error_interpolator_setup)
        return self.cache.get_value(filename, function)




def total_deviation_for_points():
    return InterpolatorTotal(sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=CONSTANTS.SCALING_VALUES).interpolated_data_for_points()


def concentration_deviation_for_points():
    return InterpolatorConcentration(sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=CONSTANTS.SCALING_VALUES).interpolated_data_for_points()

