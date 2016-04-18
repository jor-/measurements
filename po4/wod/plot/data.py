import argparse

import measurements.po4.wod.plot.interface

import util.logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot sample or interpolated mean or deviation.')

    t_dims = (1, 4, 12, 48)
    lsms = ('TMM', 'WOA13R', 'WOA13')
    parser.add_argument('data', choices=('mean', 'total_deviation', 'concentration_deviation', 'average_noise_deviation'))
    parser.add_argument('calculation', choices=('sample', 'interpolated'))
    parser.add_argument('--lsm', choices=('all', )+lsms, default='all')
    parser.add_argument('--t_dim', choices=(0,)+t_dims, type=int, default=0)

    args = parser.parse_args()

    if args.t_dim != 0:
        t_dims = (args.t_dim,)
    if args.lsm != 'all':
        lsms = (args.lsm,)

    with util.logging.Logger():
        for lsm in lsms:
            for t_dim in t_dims:
                measurements.po4.wod.plot.interface.data(args.calculation, args.data, lsm, t_dim)
