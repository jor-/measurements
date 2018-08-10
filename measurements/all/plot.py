import argparse

import matrix.constants

import util.logging

import measurements.all.data
import measurements.plot.data
import measurements.universal.correlation


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Save data from measurements.')

    parser.add_argument('--tracers', nargs='+', default=None, choices=measurements.all.data.TRACERS, help='The tracers for which the data should be plotted.')
    parser.add_argument('--min_standard_deviations', nargs='+', default=None, type=float, help='The minimal standard deviations assumed for the measurement error applied for each tracer.')
    parser.add_argument('--min_measurements_correlations', nargs='+', default=None, type=int, help='The minimal number of measurements used to calculate correlations applied to each tracer.')
    parser.add_argument('--max_box_distance_to_water', default=None, type=int, help='The maximal number of boxes allowed as distance to a water box.')
    parser.add_argument('--near_water_lsm', default='TMM', choices=measurements.all.data.LAND_SEA_MASKS, help='The land sea mask used to calculate the distances to water boxes.')

    parser.add_argument('--means_sample_lsm', action='store_true', help='Plot means for points of sample land sea mask.')
    parser.add_argument('--concentration_standard_deviations_sample_lsm', action='store_true', help='Plot concentration standard deviations for points of sample land sea mask.')
    parser.add_argument('--sample_correlation_sparsity_pattern', default=None, choices=matrix.constants.UNIVERSAL_PERMUTATION_METHODS + matrix.constants.SPARSE_ONLY_PERMUTATION_METHODS, help='Plot sparsity pattern of sample correlation of measurements with passed permutation method.')
    parser.add_argument('--sample_correlation_histogram', default=None, type=bool, choices=(True, False), help='Plot histogram of sample correlation of measurements with passed using abs.')

    parser.add_argument('--autocorrelation_sample_correlation', action='store', default=None, nargs=1, help='Plot autocorrelation of sample correlation of measurements.')
    parser.add_argument('--autocorrelation_correlation', action='store', default=None, nargs=1, help='Plot autocorrelation of correlation of measurements.')

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

        if args.sample_correlation_sparsity_pattern is not None:
            measurements.plot.data.sample_correlation_sparsity_pattern(m, permutation_method=args.sample_correlation_sparsity_pattern)

        if args.sample_correlation_histogram is not None:
            measurements.plot.data.sample_correlation_histogram(m, use_abs=args.sample_correlation_histogram)

        if args.autocorrelation_sample_correlation is not None:
            ma = measurements.universal.correlation.CorrelationCache(m)
            ma.plot_autocorrelation(axis=args.autocorrelation_sample_correlation,
                                    use_sample_correlation=True)

        if args.autocorrelation_correlation is not None:
            ma = measurements.universal.correlation.CorrelationCache(m)
            ma.plot_autocorrelation(axis=args.autocorrelation_correlation,
                                    use_sample_correlation=False)


if __name__ == "__main__":
    _main()
