import argparse

import util.logging

import measurements.all.data


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Save data from measurements.')

    parser.add_argument('--tracers', nargs='+', default=None, choices=measurements.all.data.TRACERS, help='The tracers for which the data should be saved.')
    parser.add_argument('--points_and_values', action='store_true', help='Calculate and save points and values of measurements.')
    parser.add_argument('--min_standard_deviations', nargs='+', default=None, type=float, help='The minimal standard deviations assumed for the measurement error applied for each tracer.')
    parser.add_argument('--min_measurements_correlations', nargs='+', default=None, type=float, help='The minimal number of measurements used to calculate correlations applied to each tracer.')
    parser.add_argument('--max_box_distance_to_water', default=None, type=int, help='The maximal number of boxes allowed as distance to a water box.')
    parser.add_argument('--near_water_lsm', default='TMM', help='The land sea mask used to calculate the distances to water boxes.')

    parser.add_argument('--means', action='store_true', help='Calculate and save means of measurements.')
    parser.add_argument('--concentration_standard_deviations', action='store_true', help='Calculate and save concentration standard deviations of measurements.')
    parser.add_argument('--noise_standard_deviations', action='store_true', help='Calculate and save noise standard deviations of measurements.')
    parser.add_argument('--standard_deviations', action='store_true', help='Calculate and save standard deviations of measurements.')
    parser.add_argument('--correlation', action='store_true', help='Calculate and save correlation of measurements.')

    parser.add_argument('-d', '--debug_level', default='debug', choices=util.logging.LEVELS, help='Print debug infos up to this level.')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(measurements.__version__))
    args = parser.parse_args()

    # call function
    with util.logging.Logger(level=args.debug_level):
        m = measurements.all.data.all_measurements(
            tracers=args.tracers,
            min_standard_deviations=args.min_standard_deviations,
            min_measurements_correlations=args.min_measurements_correlations,
            max_box_distance_to_water=args.max_box_distance_to_water,
            near_water_lsm=args.near_water_lsm)
        if args.points_and_values:
            m.points
            m.values
        if args.means:
            m.means
        if args.concentration_standard_deviations:
            m.concentration_standard_deviations
        if args.noise_standard_deviations:
            m.noise_standard_deviations
        if args.standard_deviations:
            m.standard_deviations
        if args.correlation:
            m.correlations_own


if __name__ == "__main__":
    _main()
