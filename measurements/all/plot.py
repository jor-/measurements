import argparse

import matrix.constants

import util.logging

import measurements.all.data
import measurements.plot.data


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Save data from measurements.')

    parser.add_argument('--tracers', nargs='+', default=None, choices=measurements.all.data.TRACERS, help='The tracers for which the data should be saved.')
    parser.add_argument('--min_standard_deviations', nargs='+', default=None, type=float, help='The minimal standard deviations assumed for the measurement error applied for each tracer.')
    parser.add_argument('--min_measurements_correlations', nargs='+', default=None, type=int, help='The minimal number of measurements used to calculate correlations applied to each tracer.')
    parser.add_argument('--max_box_distance_to_water', default=None, type=int, help='The maximal number of boxes allowed as distance to a water box.')
    parser.add_argument('--near_water_lsm', default='TMM', help='The land sea mask used to calculate the distances to water boxes.')

    parser.add_argument('--means_sample_lsm', action='store_true', help='Plot means for points of sample land sea mask.')
    parser.add_argument('--concentration_standard_deviations_sample_lsm', action='store_true', help='Plot concentration standard deviations for points of sample land sea mask.')
    parser.add_argument('--sample_correlation', action='store_true', help='Plot sample correlation of measurements.')
    parser.add_argument('--sample_correlation_sparsity_pattern', action='store_true', help='Plot sparsity pattern of sample correlation of measurements.')
    parser.add_argument('--sample_correlation_histogram', action='store_true', help='Plot histogram of sample correlation of measurements.')

    parser.add_argument('-d', '--debug_level', default='debug', choices=util.logging.LEVELS, help='Print debug infos up to this level.')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(measurements.__version__))
    args = parser.parse_args()

    tracers = args.tracers
    min_standard_deviations = args.min_standard_deviations
    min_measurements_correlations = args.min_measurements_correlations
    max_box_distance_to_water = args.max_box_distance_to_water
    near_water_lsm = args.near_water_lsm

    # call functions
    with util.logging.Logger(level=args.debug_level):
        m = measurements.all.data.all_measurements(
            tracers=tracers,
            min_standard_deviations=min_standard_deviations,
            min_measurements_correlations=min_measurements_correlations,
            max_box_distance_to_water=max_box_distance_to_water,
            near_water_lsm=near_water_lsm)

        if args.means_sample_lsm:
            measurements.plot.data.concentration_means_for_sample_lsm(m)

        if args.concentration_standard_deviations_sample_lsm:
            measurements.plot.data.concentration_standard_deviations_for_sample_lsm(m)

        if args.sample_correlation:
            for permutation_method in (matrix.constants.NO_PERMUTATION_METHOD, matrix.constants.BEST_FILL_REDUCE_PERMUTATION_METHOD):
                measurements.plot.data.sample_correlation_sparsity_pattern(m, permutation_method=permutation_method)
            for use_abs in (False, True):
                measurements.plot.data.sample_correlation_histogram(m, use_abs=use_abs)


if __name__ == "__main__":
    _main()
