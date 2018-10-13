import measurements.wod.data
import measurements.wod.po4.constants


class Measurements(measurements.wod.data.Measurements):

    def __init__(self):

        super().__init__(tracer=measurements.wod.po4.constants.TRACER)

        self.fill_strategy = measurements.wod.po4.constants.FILL_STRATEGY
