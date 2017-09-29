import measurements.wod.data
import measurements.wod.plankton.constants


class Measurements(measurements.wod.data.Measurements):

    def __init__(self,
                 tracer,
                 sample_lsm=measurements.wod.plankton.constants.SAMPLE_LSM,
                 min_standard_deviation=measurements.wod.plankton.constants.STANDARD_DEVIATION_MIN_VALUE,
                 min_measurements_correlation=measurements.wod.plankton.constants.CORRELATION_MIN_MEASUREMENTS,
                 water_lsm=None,
                 max_box_distance_to_water=None):

        if tracer not in measurements.wod.plankton.constants.TRACERS:
            raise ValueError('Tracer {} is not supported. Only {} are supported.'.format(tracer, measurements.wod.plankton.constants.TRACERS))

        if sample_lsm is None:
            sample_lsm = measurements.wod.plankton.constants.SAMPLE_LSM
        if min_standard_deviation is None:
            min_standard_deviation = measurements.wod.plankton.constants.STANDARD_DEVIATION_MIN_VALUE
        if min_measurements_correlation is None:
            min_measurements_correlation = measurements.wod.plankton.constants.CORRELATION_MIN_MEASUREMENTS

        super().__init__(
            tracer=tracer,
            sample_lsm=sample_lsm,
            min_standard_deviation=min_standard_deviation,
            min_measurements_correlation=min_measurements_correlation,
            water_lsm=water_lsm,
            max_box_distance_to_water=max_box_distance_to_water)

        self.fill_strategy = measurements.wod.plankton.constants.FILL_STRATEGY
