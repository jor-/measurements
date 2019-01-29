import measurements.wod.data
import measurements.wod.plankton.constants


class Measurements(measurements.wod.data.Measurements):

    def __init__(self, tracer, data_set_name=None):

        if tracer not in measurements.wod.plankton.constants.TRACERS:
            raise ValueError('Tracer {} is not supported. Only {} are supported.'.format(tracer, measurements.wod.plankton.constants.TRACERS))

        super().__init__(tracer, data_set_name=data_set_name)

        self.fill_strategy = measurements.wod.plankton.constants.FILL_STRATEGY
