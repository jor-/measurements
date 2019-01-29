import argparse

import util.logging

import measurements.wod.data


def _main():
    SUPPORTED_SAVE_TYPES = ('cruises_collection', 'measurements_dict', 'points_and_results')

    # parse arguments
    parser = argparse.ArgumentParser(description='Save data from world ocean database measurements.')
    parser.add_argument('tracer', choices=measurements.wod.data.SUPPORTED_TRACERS, help='The name of the tracer for which the data should be saved.')
    parser.add_argument('--data_set_name', default=None, help='The name of the data set for which the data should be saved.')
    parser.add_argument('--save_type', default='points_and_results', choices=SUPPORTED_SAVE_TYPES, help='The value type that should be saved.')
    parser.add_argument('-d', '--debug_level', default='debug', choices=util.logging.LEVELS, help='Print debug infos up to this level.')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(measurements.__version__))
    args = parser.parse_args()

    # call function
    with util.logging.Logger(level=args.debug_level):
        if args.save_type == 'cruises_collection':
            measurements.wod.data.cruises_collection(args.tracer, args.data_set_name)
        elif args.save_type == 'measurements_dict':
            measurements.wod.data.measurements_dict(args.tracer, args.data_set_name)
        elif args.save_type == 'points_and_results':
            measurements.wod.data.points_and_results(args.tracer, args.data_set_name)


if __name__ == "__main__":
    _main()
