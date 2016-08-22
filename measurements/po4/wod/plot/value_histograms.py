from util.logging import Logger

import measurements.po4.wod.data.values
import measurements.land_sea_mask.lsm
import measurements.util.plot

with Logger():
    lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R(t_dim=48)
    file = '/tmp/wod13_{}_value_histogram_{}_{}.png'.format('{}', lsm, '{}')
    min_measurements = 1000
    step_size = 0.05

    m = measurements.po4.wod.data.values.measurement_dict()
    measurements.util.plot.value_histograms(m, lsm=lsm, file=file.format('po4', '{}'), min_measurements=min_measurements, step_size=step_size)

    m = measurements.po4.wod.data.values.measurement_dict()
    m.set_min_result(0.01)
    m.log_results()
    measurements.util.plot.value_histograms(m, lsm=lsm, file=file.format('log_po4', '{}'), min_measurements=min_measurements, step_size=step_size)


