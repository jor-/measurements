import argparse

import matrix.constants

import util.logging

import measurements.all.data
import measurements.plot.data
import measurements.universal.correlation


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Save data from measurements.')

    parser.add_argument('--tracers', nargs='+', choices=measurements.all.data.TRACERS, default=measurements.all.data.TRACERS, help='The tracers for which the data should be saved.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=None, help='The maximal number of boxes allowed as distance to a water box.')
    parser.add_argument('--water_lsm', choices=measurements.all.data.LAND_SEA_MASKS, default='TMM', help='The land sea mask used to calculate the distances to water boxes.')

    parser.add_argument('--min_measurements_mean', type=int, default=None, help='The minimal number of measurements used to calculate means applied to each tracer.')
    parser.add_argument('--min_measurements_quantile', type=int, nargs='+', default=None, help='The minimal number of measurements used to calculate quantiles applied to each tracer.')
    parser.add_argument('--min_measurements_standard_deviation', type=int, default=None, help='The minimal number of measurements used to calculate standard deviations applied to each tracer.')
    parser.add_argument('--min_measurements_correlation', type=int, nargs='+', default=None, help='The minimal number of measurements used to calculate correlations applied to each tracer.')

    parser.add_argument('--min_standard_deviation', nargs='+', default=None, type=float, help='The minimal standard deviations assumed for the measurement error applied for each tracer.')

    parser.add_argument('--means_sample_lsm', action='store_true', help='Plot means for points of sample land sea mask.')
    parser.add_argument('--quantiles_sample_lsm', type=float, default=None, help='Plot passed quantiles for points of sample land sea mask.')
    parser.add_argument('--concentration_standard_deviations_sample_lsm', action='store_true', help='Plot concentration standard deviations for points of sample land sea mask.')
    parser.add_argument('--standard_deviations_sample_lsm', action='store_true', help='Plot standard deviations for points of sample land sea mask.')
    parser.add_argument('--interquartile_range_for_sample_lsm', action='store_true', help='Plot interquartile range for points of sample land sea mask.')
    parser.add_argument('--relative_standard_deviations_for_sample_lsm', action='store_true', help='Plot relative standard deviations for points of sample land sea mask.')
    parser.add_argument('--quartile_coefficient_of_dispersion_for_sample_lsm', action='store_true', help='Plot quartile coefficient of dispersion for points of sample land sea mask.')

    parser.add_argument('--sample_correlation_sparsity_pattern', choices=matrix.constants.UNIVERSAL_PERMUTATION_METHODS + matrix.constants.SPARSE_ONLY_PERMUTATION_METHODS, default=None, help='Plot sparsity pattern of sample correlation of measurements with passed permutation method.')
    parser.add_argument('--sample_correlation_histogram', type=bool, choices=(True, False), default=None, help='Plot histogram of sample correlation of measurements with passed using abs.')
    parser.add_argument('--sample_correlation_correlations', action='store', default=None, nargs='*', help='Plot correlation of sample correlation of measurements.')
    parser.add_argument('--sample_correlation_autocorrelations', action='store', default=None, nargs='*', help='Plot autocorrelation of sample correlation of measurements.')
    parser.add_argument('--sample_correlation_violin_autocorrelations', action='store', default=None, nargs='*', help='Plot autocorrelation of sample correlation of measurements as violin plot.')

    parser.add_argument('--correlation_sparsity_pattern', action='store_true', help='Plot sparsity pattern of correlation and sample correlation of measurements in one plot.')
    parser.add_argument('--correlation_correlations', action='store', default=None, nargs='*', help='Plot correlation of correlation of measurements.')
    parser.add_argument('--correlation_autocorrelations', action='store', default=None, nargs='*', help='Plot autocorrelation of correlation of measurements.')
    parser.add_argument('--correlation_violin_autocorrelations', action='store', default=None, nargs='*', help='Plot autocorrelation of correlation of measurements as violin plot.')

    parser.add_argument('-d', '--debug_level', default='debug', choices=util.logging.LEVELS, help='Print debug infos up to this level.')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(measurements.__version__))
    args = parser.parse_args()

    # call functions
    with util.logging.Logger(level=args.debug_level):
        # init measurements object
        m = measurements.all.data.all_measurements(
            tracers=args.tracers,
            min_measurements_mean=args.min_measurements_mean,
            min_measurements_quantile=args.min_measurements_quantile,
            min_measurements_standard_deviation=args.min_measurements_standard_deviation,
            min_measurements_correlation=args.min_measurements_correlation,
            min_standard_deviation=args.min_standard_deviation,
            max_box_distance_to_water=args.max_box_distance_to_water,
            water_lsm=args.water_lsm)

        try:
            measurements_list = m.measurements_list
        except AttributeError:
            measurements_list = [m]

        # plot
        for mi in measurements_list:
            if args.means_sample_lsm:
                measurements.plot.data.means_for_sample_lsm(mi)

            if args.quantiles_sample_lsm:
                measurements.plot.data.quantiles_for_sample_lsm(mi, args.quantiles_sample_lsm)

            if args.concentration_standard_deviations_sample_lsm:
                measurements.plot.data.concentration_standard_deviations_for_sample_lsm(mi)

            if args.standard_deviations_sample_lsm:
                measurements.plot.data.standard_deviations_for_sample_lsm(mi)

            if args.interquartile_range_for_sample_lsm:
                measurements.plot.data.interquartile_range_for_sample_lsm(mi)

            if args.relative_standard_deviations_for_sample_lsm:
                measurements.plot.data.relative_standard_deviations_for_sample_lsm(mi)

            if args.quartile_coefficient_of_dispersion_for_sample_lsm:
                measurements.plot.data.quartile_coefficient_of_dispersion_for_sample_lsm(mi)

        if args.sample_correlation_sparsity_pattern is not None:
            measurements.plot.data.sample_correlation_sparsity_pattern(m, permutation_method=args.sample_correlation_sparsity_pattern)

        if args.sample_correlation_histogram is not None:
            measurements.plot.data.sample_correlation_histogram(m, use_abs=args.sample_correlation_histogram)

        if args.correlation_sparsity_pattern:
            measurements.plot.data.correlation_and_sample_correlation_sparsity_pattern(m)

        mc = measurements.universal.correlation.CorrelationCache(m)
        if args.sample_correlation_correlations is not None:
            mc.plot_correlation(axis=args.sample_correlation_correlations,
                                use_sample_correlation=True)
        if args.sample_correlation_autocorrelations is not None:
            mc.plot_autocorrelation(axis=args.sample_correlation_autocorrelations,
                                    use_sample_correlation=True)
        if args.sample_correlation_violin_autocorrelations is not None:
            mc.plot_violin_autocorrelation(axis=args.sample_correlation_violin_autocorrelations,
                                           use_sample_correlation=True)
        if args.correlation_correlations is not None:
            mc.plot_correlation(axis=args.correlation_correlations,
                                use_sample_correlation=False)
        if args.correlation_autocorrelations is not None:
            mc.plot_autocorrelation(axis=args.correlation_autocorrelations,
                                    use_sample_correlation=False)
        if args.correlation_violin_autocorrelations is not None:
            mc.plot_violin_autocorrelation(axis=args.correlation_violin_autocorrelations,
                                           use_sample_correlation=False)


if __name__ == "__main__":
    _main()
