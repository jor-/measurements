import measurements.wod.data
import measurements.wod.po4.constants


class Measurements(measurements.wod.data.Measurements):

    def __init__(self,
                 sample_lsm=measurements.wod.po4.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.wod.po4.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.wod.po4.constants.CORRELATION_MIN_MEASUREMENTS,
                 water_lsm=None,
                 max_box_distance_to_water=None):

        if sample_lsm is None:
            sample_lsm = measurements.wod.po4.constants.SAMPLE_LSM
        if min_standard_deviation is None:
            min_standard_deviation = measurements.wod.po4.constants.STANDARD_DEVIATION_MIN_VALUE
        if min_measurements_correlation is None:
            min_measurements_correlation = measurements.wod.po4.constants.CORRELATION_MIN_MEASUREMENTS

        super().__init__(
            tracer=measurements.wod.po4.constants.TRACER,
            sample_lsm=sample_lsm,
            min_standard_deviation=min_standard_deviation,
            min_measurements_correlation=min_measurements_correlation,
            water_lsm=water_lsm,
            max_box_distance_to_water=max_box_distance_to_water)

        self.fill_strategy = measurements.wod.po4.constants.FILL_STRATEGY
