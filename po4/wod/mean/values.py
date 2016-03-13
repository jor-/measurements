import measurements.po4.wod.data.interpolate

import measurements.po4.wod.mean.constants as CONSTANTS


DEFAULT_INTERPOLATOR_SETUPS = {
'concentration': {2: {'lsm_12_woa13r':(0.1,6,0.6,1)}, 3: {'lsm_12_woa13r':(0.2,3,0.6,1)}},
}


class InterpolatorDirect(measurements.po4.wod.data.interpolate.Interpolator):

    def __init__(self, min_values=CONSTANTS.MIN_MEASUREMENTS, sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=None):
        self.min_values = min_values
        filename = CONSTANTS.INTERPOLATED_MEAN_DIRECT_FILENAME.format(min_values=min_values, sample_lsm=sample_lsm, scaling_values='{scaling_values}', points='{points}', interpolator_setup='{interpolator_setup}')
        super().__init__(CONSTANTS.MEAN_DIR, filename, sample_lsm=sample_lsm, scaling_values=scaling_values)
    

    @property
    def data(self):
        m = measurements.po4.wod.data.values.measurement_dict()
        m.categorize_indices_to_lsm(self.sample_lsm, discard_year=False)
        m.discard_year()
        m.means(min_values=self.min_values, return_type='self')
        return m.items()




class InterpolatorConcentration(measurements.po4.wod.data.interpolate.Interpolator):

    def __init__(self, min_values=CONSTANTS.MIN_MEASUREMENTS, sample_lsm=CONSTANTS.SAMPLE_LSM, scaling_values=None):
        self.min_values = min_values
        filename = CONSTANTS.INTERPOLATED_MEAN_CONCENTRATION_FILENAME.format(min_values=min_values, sample_lsm=sample_lsm, scaling_values='{scaling_values}', points='{points}', interpolator_setup='{interpolator_setup}')
        try:
            default_interpolator_setups = DEFAULT_INTERPOLATOR_SETUPS['concentration'][min_values]
        except KeyError:
            default_interpolator_setups = None
        super().__init__(CONSTANTS.MEAN_DIR, filename, sample_lsm=sample_lsm, scaling_values=scaling_values, default_interpolator_setups=default_interpolator_setups)
    

    @property
    def data(self):
        m = measurements.po4.wod.data.values.measurement_dict()
        m.categorize_indices_to_lsm(self.sample_lsm, discard_year=False)
        m.means(min_values=1, return_type='self')
        m.discard_year()
        m.means(min_values=self.min_values, return_type='self')
        return m.items()


