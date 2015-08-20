import argparse

import measurements.po4.wod.mean.values
import measurements.po4.wod.deviation.values

import util.logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save interpolated data for points or lsm.')

    parser.add_argument('--aim', choices=('points', 'TMM', 'WOA13', 'WOA13R'), help='The aim to interpolate to.')
    parser.add_argument('--data', choices=('mean', 'deviation'), help='The data to interpolate.')

    args = vars(parser.parse_args())
    aim_kind = args['aim']
    data_kind = args['data']

    with util.logging.Logger():
        if data_kind == 'mean':
            interpolator = measurements.po4.wod.mean.values.Interpolator()
        elif data_kind == 'deviation':
            interpolator = measurements.po4.wod.deviation.values.Interpolator()

        if aim_kind == 'points':
            interpolator.data_for_points()
        elif aim_kind == 'TMM':
            interpolator.data_for_TMM(t_dim=48)
        elif aim_kind == 'WOA13':
            interpolator.data_for_WOA13(t_dim=48)
        elif aim_kind == 'WOA13R':
            interpolator.data_for_WOA13R(t_dim=48)

