import measurements.wod.data
import measurements.wod.constants
import measurements.wod.po4.constants
import measurements.universal.data


class Measurements(measurements.wod.data.Measurements):

    def __init__(self,
                 sample_lsm=measurements.wod.po4.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.wod.po4.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.wod.po4.constants.CORRELATION_MIN_MEASUREMENTS):

        super().__init__(
            sample_lsm,
            tracer=measurements.wod.po4.constants.TRACER,
            data_set_name=measurements.wod.constants.DATA_SET_NAME,
            min_standard_deviation=min_standard_deviation,
            min_measurements_correlation=min_measurements_correlation)

        self.fill_strategy = measurements.wod.po4.constants.FILL_STRATEGY


class MeasurementsNearWater(measurements.universal.data.MeasurementsAnnualPeriodicNearWaterCache):

    def __init__(self,
                 water_lsm=None,
                 max_box_distance_to_water=0,
                 sample_lsm=measurements.wod.po4.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.wod.po4.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.wod.po4.constants.CORRELATION_MIN_MEASUREMENTS):

        measurements = Measurements(sample_lsm=sample_lsm,
                                    min_standard_deviation=min_standard_deviation,
                                    min_measurements_correlation=min_measurements_correlation)

        super().__init__(measurements, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water)
