import argparse

import util.logging

import measurements.all.data
import measurements.universal.constants
import measurements.universal.correlation


def _main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Save data from measurements.')

    parser.add_argument('--tracers', nargs='+', choices=measurements.all.data.TRACERS, default=None, help='The tracers for which the data should be saved.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=None, help='The maximal number of boxes allowed as distance to a water box.')
    parser.add_argument('--water_lsm', choices=measurements.all.data.LAND_SEA_MASKS, default='TMM', help='The land sea mask used to calculate the distances to water boxes.')

    parser.add_argument('--min_measurements_mean', type=int, default=None, help='The minimal number of measurements used to calculate means applied to each tracer.')
    parser.add_argument('--min_measurements_quantile', type=int, nargs='+', default=None, help='The minimal number of measurements used to calculate quantiles applied to each tracer.')
    parser.add_argument('--min_measurements_standard_deviation', type=int, default=None, help='The minimal number of measurements used to calculate standard deviations applied to each tracer.')
    parser.add_argument('--min_measurements_correlation', type=int, nargs='+', default=None, help='The minimal number of measurements used to calculate correlations applied to each tracer.')

    parser.add_argument('--min_standard_deviation', nargs='+', default=None, type=float, help='The minimal standard deviations assumed for the measurement error applied for each tracer.')

    parser.add_argument('--points_and_values', action='store_true', help='Calculate and save points and values of measurements.')

    parser.add_argument('--means', action='store_true', help='Calculate and save means for measurement points.')
    parser.add_argument('--quantiles', action='store', type=float, help='Calculate and save the passed quantile for measurement points.')
    parser.add_argument('--concentration_standard_deviations', action='store_true', help='Calculate and save concentration standard deviations for measurement points.')
    parser.add_argument('--noise_standard_deviations', action='store_true', help='Calculate and save noise standard deviations of measurements.')
    parser.add_argument('--standard_deviations', action='store_true', help='Calculate and save standard deviations of measurements.')

    parser.add_argument('--means_sample_lsm', action='store_true', help='Calculate and save means for points of sample land sea mask.')
    parser.add_argument('--quantiles_sample_lsm', action='store', type=float, help='Calculate and save the passed quantile for points of sample land sea mask.')
    parser.add_argument('--concentration_standard_deviations_sample_lsm', action='store_true', help='Calculate and save concentration standard deviations for points of sample land sea mask.')
    parser.add_argument('--average_noise_standard_deviations_for_sample_lsm', action='store_true', help='Calculate and save average noise standard deviations for points of sample land sea mask.')
    parser.add_argument('--standard_deviations_sample_lsm', action='store_true', help='Calculate and save standard deviations for points of sample land sea mask.')

    parser.add_argument('--sample_correlation', action='store_true', help='Calculate and save sample correlation of measurements.')
    parser.add_argument('--correlation', action='store_true', help='Calculate and save correlation of measurements.')
    parser.add_argument('--sample_correlation_autocorrelation', action='store', nargs='*', help='Calculate and save autocorrelation of sample correlation of measurements.')
    parser.add_argument('--correlation_autocorrelation', action='store', nargs='*', help='Calculate and save autocorrelation of correlation of measurements.')

    parser.add_argument('-d', '--debug_level', default='debug', choices=util.logging.LEVELS, help='Print debug infos up to this level.')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(measurements.__version__))
    args = parser.parse_args()

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

        # calculate
        if args.points_and_values:
            m.points
            m.values
        if args.means:
            m.means
        if args.quantiles is not None:
            m.quantiles(args.quantiles)
        if args.concentration_standard_deviations:
            m.concentration_standard_deviations
        if args.noise_standard_deviations:
            m.noise_standard_deviations
        if args.standard_deviations:
            m.standard_deviations
        if args.sample_correlation:
            m.correlations_own_sample_matrix
        if args.correlation:
            m.correlations_own

        try:
            m_list = m.measurements_list
        except AttributeError:
            m_list = [m]

        for m in m_list:
            if args.means_sample_lsm:
                m.means_for_sample_lsm()
            if args.quantiles_sample_lsm is not None:
                m.quantiles_for_sample_lsm(args.quantiles_sample_lsm)
            if args.concentration_standard_deviations_sample_lsm:
                m.concentration_standard_deviations_for_sample_lsm()
            if args.average_noise_standard_deviations_for_sample_lsm:
                m.average_noise_standard_deviations_for_sample_lsm()
            if args.standard_deviations_sample_lsm:
                m.standard_deviations_for_sample_lsm()

        if args.sample_correlation_autocorrelation is not None:
            if len(args.sample_correlation_autocorrelation) == 0:
                sample_correlation_autocorrelation = None
            else:
                sample_correlation_autocorrelation = args.sample_correlation_autocorrelation
            ma = measurements.universal.correlation.CorrelationCache(m)
            ma.autocorrelation_array(axis=sample_correlation_autocorrelation,
                                     use_sample_correlation=True)

        if args.correlation_autocorrelation is not None:
            if len(args.correlation_autocorrelation) == 0:
                correlation_autocorrelation = None
            else:
                correlation_autocorrelation = args.correlation_autocorrelation
            ma = measurements.universal.correlation.CorrelationCache(m)
            ma.autocorrelation_array(axis=correlation_autocorrelation,
                                     use_sample_correlation=False)


if __name__ == "__main__":
    _main()
