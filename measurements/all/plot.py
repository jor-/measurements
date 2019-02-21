import argparse

import matplotlib.pyplot as plt

import matrix.constants

import util.logging

import measurements.all.data
import measurements.plot.constants
import measurements.plot.data
import measurements.plot.number_of_measurements
import measurements.universal.correlation


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Save data from measurements.')

    # tracer and points
    parser.add_argument('--tracers', choices=measurements.all.data.TRACERS, default=measurements.all.data.TRACERS, nargs='+', help='The tracers for which the data should be saved.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=None, help='The maximal number of boxes allowed as distance to a water box.')
    parser.add_argument('--water_lsm', choices=measurements.all.data.LAND_SEA_MASKS, default='TMM', help='The land sea mask used to calculate the distances to water boxes.')

    # min measurements
    parser.add_argument('--min_measurements_mean', type=int, default=None, nargs='+', help='The minimal number of measurements used to calculate means applied to each tracer.')
    parser.add_argument('--min_measurements_quantile', type=int, default=None, nargs='+', help='The minimal number of measurements used to calculate quantiles applied to each tracer.')
    parser.add_argument('--min_measurements_standard_deviation', type=int, default=None, nargs='+', help='The minimal number of measurements used to calculate standard deviations applied to each tracer.')
    parser.add_argument('--min_measurements_correlation', type=int, default=None, nargs='+', help='The minimal number of measurements used to calculate correlations applied to each tracer.')

    # min values
    parser.add_argument('--min_standard_deviation', type=float, default=None, nargs='+', help='The minimal standard deviations assumed for the measurement error applied for each tracer.')

    # number of measurements
    parser.add_argument('--number_of_measurements_per_time', type=float, help='Plots the number of measurements per passed time step.')
    parser.add_argument('--number_of_measurements_per_year', type=int, help='Plots the number of measurements within a year per passed number of bins.')
    parser.add_argument('--number_of_measurements_per_depth', type=float, help='Plots the number of measurements per passed depth step.')
    parser.add_argument('--number_of_measurements_per_space', action='store_true', help='Plots the number of measurements per sample land sea mask grid summed up.')
    parser.add_argument('--number_of_measurements_per_space_each_depth', action='store_true', help='Plots the number of measurements per sample land sea mask grid.')

    # expectation values
    parser.add_argument('--means_sample_lsm', action='store_true', help='Plot means for points of sample land sea mask.')
    parser.add_argument('--concentration_quantiles_sample_lsm', type=float, help='Plot passed quantiles for points of sample land sea mask.')

    # spread values
    parser.add_argument('--concentration_standard_deviations_sample_lsm', action='store_true', help='Plot concentration standard deviations for points of sample land sea mask.')
    parser.add_argument('--average_noise_standard_deviations_sample_lsm', action='store_true', help='Plot average noise standard deviations for points of sample land sea mask.')
    parser.add_argument('--standard_deviations_sample_lsm', action='store_true', help='Plot standard deviations for points of sample land sea mask.')
    parser.add_argument('--concentration_relative_standard_deviations_sample_lsm', action='store_true', help='Plot concentration relative standard deviations for points of sample land sea mask.')
    parser.add_argument('--relative_standard_deviations_sample_lsm', action='store_true', help='Plot relative standard deviations for points of sample land sea mask.')

    parser.add_argument('--concentration_interquartile_range_sample_lsm', action='store_true', help='Plot concentration interquartile range for points of sample land sea mask.')
    parser.add_argument('--average_noise_interquartile_range_sample_lsm', action='store_true', help='Plot average noise interquartile range for points of sample land sea mask.')
    parser.add_argument('--concentration_quartile_coefficient_of_dispersion_sample_lsm', action='store_true', help='Plot quartile coefficient of dispersion for points of sample land sea mask.')

    # correlation
    parser.add_argument('--correlation_histogram', action='store_true', help='Plot histogram of correlation of measurements with passed using abs.')
    parser.add_argument('--correlation_sparsity_pattern', choices=matrix.constants.UNIVERSAL_PERMUTATION_METHODS + matrix.constants.SPARSE_ONLY_PERMUTATION_METHODS + ('default',), nargs='?', const='default', help='Plot sparsity pattern of  correlation of measurements with passed permutation method.')
    parser.add_argument('--correlation_and_sample_correlation_sparsity_pattern', action='store_true', help='Plot sparsity pattern of correlation and sample correlation of measurements in one plot.')
    parser.add_argument('--correlation_means', action='store', nargs='+', help='Plot average correlations of correlation of measurements for passed axis.')
    parser.add_argument('--correlation_standard_deviations', action='store', nargs='+', help='Plot standard deviations of correlation of measurements for passed axis.')
    parser.add_argument('--correlation_inter_quartile_ranges', action='store', nargs='+', help='Plot inter quantile ranges of correlation of measurements for passed axis.')
    parser.add_argument('--correlation_lag_means', action='store', nargs='+', help='Plot average of correlation_lags of measurements for passed axis.')
    parser.add_argument('--correlation_lag_standard_deviations', action='store', nargs='+', help='Plot standard deviations of correlation_lags of measurements for passed axis.')
    parser.add_argument('--correlation_lag_inter_quartile_ranges', action='store', nargs='+', help='Plot inter quantile ranges of correlation_lags of measurements for passed axis.')
    parser.add_argument('--correlation_lag_violins', action='store', nargs='+', help='Plot correlation_lags of correlation of measurements as violin plot for passed axis.')
    parser.add_argument('--use_sample_correlation', action='store_true', help='Use sample correlation instead of correlation for plots.')
    parser.add_argument('--use_abs', action='store_true', help='Use abs values for supported correlation plots.')

    # overwrite
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')

    # file extension and backend
    parser.add_argument('--file_extension', default=measurements.plot.constants.PLOT_DEFAULT_FILE_EXTENSION, help='The file extension that should be used to store the plot.')
    parser.add_argument('--backend', help='The plot backend matplotlib should use.')

    # debug
    parser.add_argument('-d', '--debug_level', default='debug', choices=util.logging.LEVELS, help='Print debug infos up to this level.')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(measurements.__version__))
    args = parser.parse_args()

    # set file extension
    measurements.plot.constants.PLOT_DEFAULT_FILE_EXTENSION = args.file_extension

    # set backend
    if args.backend is not None:
        plt.switch_backend(args.backend)

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

            # number of measurements_i

            if args.number_of_measurements_per_time:
                measurements.plot.number_of_measurements.per_time(
                    mi, step_size=args.number_of_measurements_per_time,
                    overwrite=args.overwrite)

            if args.number_of_measurements_per_year:
                measurements.plot.number_of_measurements.per_year(
                    mi, number_of_bins=args.number_of_measurements_per_year,
                    overwrite=args.overwrite)

            if args.number_of_measurements_per_depth:
                measurements.plot.number_of_measurements.per_depth(
                    mi, step_size=args.number_of_measurements_per_depth,
                    overwrite=args.overwrite)

            if args.number_of_measurements_per_space_each_depth:
                for max_value_fixed in (True, False):
                    measurements.plot.number_of_measurements.per_space_each_depth(
                        mi, max_value_fixed=max_value_fixed,
                        overwrite=args.overwrite)

            if args.number_of_measurements_per_space:
                measurements.plot.number_of_measurements.per_space(
                    mi, overwrite=args.overwrite)

            # expectation values

            if args.means_sample_lsm:
                measurements.plot.data.means_for_sample_lsm(
                    mi, overwrite=args.overwrite)

            if args.concentration_quantiles_sample_lsm:
                measurements.plot.data.concentration_quantiles_for_sample_lsm(
                    mi, args.concentration_quantiles_sample_lsm,
                    min_measurements=args.min_measurements_quantile,
                    overwrite=args.overwrite)

            # spread values

            if args.concentration_standard_deviations_sample_lsm:
                measurements.plot.data.concentration_standard_deviations_for_sample_lsm(
                    mi, overwrite=args.overwrite)

            if args.average_noise_standard_deviations_sample_lsm:
                measurements.plot.data.average_noise_standard_deviations_for_sample_lsm(
                    mi, overwrite=args.overwrite)

            if args.standard_deviations_sample_lsm:
                measurements.plot.data.standard_deviations_for_sample_lsm(
                    mi, overwrite=args.overwrite)

            if args.concentration_relative_standard_deviations_sample_lsm:
                measurements.plot.data.concentration_relative_standard_deviations_for_sample_lsm(
                    mi, overwrite=args.overwrite)

            if args.relative_standard_deviations_sample_lsm:
                measurements.plot.data.relative_standard_deviations_for_sample_lsm(
                    mi, overwrite=args.overwrite)

            if args.concentration_interquartile_range_sample_lsm:
                measurements.plot.data.concentration_interquartile_range_for_sample_lsm(
                    mi, min_measurements=args.min_measurements_quantile, overwrite=args.overwrite)

            if args.average_noise_interquartile_range_sample_lsm:
                measurements.plot.data.average_noise_interquartile_range_for_sample_lsm(
                    mi, min_measurements=args.min_measurements_quantile, overwrite=args.overwrite)

            if args.concentration_quartile_coefficient_of_dispersion_sample_lsm:
                measurements.plot.data.concentration_quartile_coefficient_of_dispersion_for_sample_lsm(
                    mi, min_measurements=args.min_measurements_quantile, overwrite=args.overwrite)

        # sample correlation and correlation

        if args.correlation_histogram:
            measurements.plot.data.correlation_histogram(
                m, use_abs=args.use_abs,
                use_sample_correlation=args.use_sample_correlation,
                overwrite=args.overwrite)

        if args.correlation_sparsity_pattern:
            if args.correlation_sparsity_pattern == 'default':
                permutation_method = None
            else:
                permutation_method = args.correlation_sparsity_pattern
            measurements.plot.data.correlation_sparsity_pattern(
                m, permutation_method=permutation_method,
                use_sample_correlation=args.use_sample_correlation,
                overwrite=args.overwrite)

        if args.correlation_and_sample_correlation_sparsity_pattern:
            measurements.plot.data.correlation_and_sample_correlation_sparsity_pattern(
                m, overwrite=args.overwrite)

        mc = measurements.universal.correlation.CorrelationCache(m)
        if args.correlation_means:
            mc.plot_correlation(axis=args.correlation_means,
                                plot_type='means',
                                use_abs=args.use_abs,
                                use_sample_correlation=args.use_sample_correlation,
                                overwrite=args.overwrite)
        if args.correlation_standard_deviations:
            mc.plot_correlation(axis=args.correlation_standard_deviations,
                                plot_type='standard_deviations',
                                use_abs=args.use_abs,
                                use_sample_correlation=args.use_sample_correlation,
                                overwrite=args.overwrite)
        if args.correlation_inter_quartile_ranges:
            mc.plot_correlation(axis=args.correlation_inter_quartile_ranges,
                                plot_type='inter_quartile_ranges',
                                use_abs=args.use_abs,
                                use_sample_correlation=args.use_sample_correlation,
                                overwrite=args.overwrite)
        if args.correlation_lag_means:
            mc.plot_correlation_lag(axis=args.correlation_lag_means,
                                    plot_type='means',
                                    use_abs=args.use_abs,
                                    use_sample_correlation=args.use_sample_correlation,
                                    overwrite=args.overwrite)
        if args.correlation_lag_standard_deviations:
            mc.plot_correlation_lag(axis=args.correlation_lag_standard_deviations,
                                    plot_type='standard_deviations',
                                    use_abs=args.use_abs,
                                    use_sample_correlation=args.use_sample_correlation,
                                    overwrite=args.overwrite)
        if args.correlation_lag_inter_quartile_ranges:
            mc.plot_correlation_lag(axis=args.correlation_lag_inter_quartile_ranges,
                                    plot_type='inter_quartile_ranges',
                                    use_abs=args.use_abs,
                                    use_sample_correlation=args.use_sample_correlation,
                                    overwrite=args.overwrite)
        if args.correlation_lag_violins:
            mc.plot_correlation_lag_violins(axis=args.correlation_lag_violins,
                                            use_sample_correlation=args.use_sample_correlation,
                                            overwrite=args.overwrite)


if __name__ == "__main__":
    _main()
