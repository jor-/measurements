import numpy as np

import measurements.util.correlation
import measurements.po4.wod.correlation.estimation

# from measurements.po4.constants import CORRELATION_QUANTITY_SAME_BOX, CORRELATION_SAME_BOX, CORRELATION_MIN_ABS_VALUE, CORRELATION_MAX_ABS_VALUE


class CorrelationModel(measurements.util.correlation.Model):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.sample_value_estimation = measurements.po4.wod.correlation.estimation.SampleCorrelation(min_values=self.min_values, max_year_diff=self.max_year_diff, same_box_quantity=self.same_box_quantity, same_box_correlation=self.same_box_correlation, no_data_correlation=self.no_data_correlation, return_type=self.return_type)


    def sample_value(self, keys):
        return self.sample_value_estimation[keys]

    @property
    def number_of_sample_values(self):
        return self.sample_value_estimation.number_of_sample_values

    @property
    def effective_max_year_diff(self):
        return self.sample_value_estimation.effective_max_year_diff








# import numpy as np
#
# import measurements.po4.wod.correlation.estimation
#
# from measurements.po4.constants import CORRELATION_SAME_BOX, CORRELATION_MIN_ABS_VALUE, CORRELATION_MAX_ABS_VALUE
#
#
# class CorrelationModel():
#
#     def __init__(self, min_values=5, max_year_diff=1, same_box_correlation=CORRELATION_SAME_BOX, min_abs_correlation=CORRELATION_MIN_ABS_VALUE, max_abs_correlation=CORRELATION_MAX_ABS_VALUE):
#         self.sample_correlation = measurements.po4.wod.correlation.estimation.SampleCorrelation(min_values=min_values, max_year_diff=max_year_diff, same_box_correlation=same_box_correlation, no_data_value=0)
#         self.min_abs_correlation = min_abs_correlation
#         self.max_abs_correlation = max_abs_correlation
#
#     #
#     # def value(self, keys):
#     #     try:
#     #         correlation = self.sample_correlation[keys]
#     #     except KeyError:
#     #         correlation = 0
#     #
#     #     if np.abs(correlation) < self.min_abs_correlation:
#     #         return 0
#     #     else:
#     #         correlation = np.sign(correlation) * min(np.abs(correlation), self.max_abs_correlation)
#     #         return correlation
#
#     def value(self, keys):
#         correlation = self.sample_correlation[keys]
#
#         if np.abs(correlation) < self.min_abs_correlation:
#             correlation = 0
#         else:
#             correlation = np.sign(correlation) * min(np.abs(correlation), self.max_abs_correlation)
#             return correlation
#
#
#     def __getitem__(self, key):
#         return self.value(key)
#
#
#     @property
#     def number_of_sample_values(self):
#         return self.sample_correlation.number_of_sample_values
#
#     def __len__(self):
#         return self.number_of_sample_values
#
#
#     @property
#     def effective_max_year_diff(self):
#         return self.sample_correlation.effective_max_year_diff
#