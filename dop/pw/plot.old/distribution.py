from util.logging import Logger
import measurements.dop.pw.data
import measurements.util.plot

with Logger():
    measurements.util.plot.distribution(measurements.dop.pw.data.measurement_dict(), file='/tmp/dop_pw_distribution.png', year_len=1, use_log_norm=False)