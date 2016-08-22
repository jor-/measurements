import argparse

import measurements.dop.pw.data
import measurements.po4.wod.data.values
import measurements.land_sea_mask.lsm
import measurements.util.plot

import util.logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting interpolation data.')

    types = ('time', 'year', 'depth', 'space', 'space_tmm', 'space_woa13r', 'space_woa13')
    parser.add_argument('data', choices=('dop', 'po4'), help='The data to plot.')
    parser.add_argument('--type', choices=('all',)+types, default='all', help='The type of the plot.')

    args = parser.parse_args()

    data = args.data

    if data == 'dop':
        load_data = measurements.dop.pw.data.measurement_dict
        tick_power = None
        use_log_scale = False
    else:
        load_data = measurements.po4.wod.data.values.measurement_dict
        data = 'po4_-_wod13'
        tick_power = 3
        use_log_scale = True

    t_min = measurements.po4.wod.data.values.points()[:,0].min()
    t_max = measurements.po4.wod.data.values.points()[:,0].max()
    z_max = measurements.po4.wod.data.values.points()[:,3].max()

    type = args.type
    if type != 'all':
        types = (type,)


    with util.logging.Logger():
        for type in types:

            if type == 'time':
                measurements.util.plot.distribution_time(load_data(), file='/tmp/{}_-_distribution_time_-.png'.format(data), t_min=t_min, t_max=t_max, tick_power=tick_power)
            if type == 'year':
                for time_len in (365, 2880, 48):
                    measurements.util.plot.distribution_year(load_data(), file='/tmp/{}_-_distribution_year_{}_-.png'.format(data, time_len), time_step=1./time_len, tick_power=tick_power)
            if type == 'depth':
                measurements.util.plot.distribution_depth(load_data(), file='/tmp/{}_-_distribution_depth.png_-'.format(data), z_max=z_max, use_log_scale=use_log_scale)
            if type[:5] == 'space':
                type = type[6:]
                lsms = []
                if type == 'tmm' or type == '':
                    lsms.append(measurements.land_sea_mask.lsm.LandSeaMaskTMM())
                if type == 'woa13r' or type == '':
                    lsms.append(measurements.land_sea_mask.lsm.LandSeaMaskWOA13R())
                if type == 'woa13' or type == '':
                    lsms.append(measurements.land_sea_mask.lsm.LandSeaMaskWOA13())

                for lsm in lsms:
                    for t_dim in (12, 4, 1):
                        lsm.t_dim = t_dim
                        measurements.util.plot.distribution_space(load_data(), lsm=lsm, file='/tmp/{}_-_distribution_space_-.png'.format(data), use_log_scale=use_log_scale)

