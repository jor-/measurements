from util.logging import Logger
import measurements.dop.pw.data
import measurements.util.plot

with Logger():
    measurements.util.plot.distribution_year(measurements.dop.pw.data.measurement_dict(), file='/tmp/wod_po4_distribution_year.png', line_width=4, spine_line_width=2)