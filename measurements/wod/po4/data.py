import measurements.wod.data
import measurements.wod.po4.constants


class Measurements(measurements.wod.data.Measurements):

    def __init__(self, data_set_name=None):

        super().__init__(measurements.wod.po4.constants.TRACER, data_set_name)

        self.fill_strategy = measurements.wod.po4.constants.FILL_STRATEGY
