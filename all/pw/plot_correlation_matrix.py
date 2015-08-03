import argparse

import measurements.all.pw.plot

import util.logging

if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--min_values', type=int)
    parser.add_argument('-y', '--max_year_diff', type=int, default=-1)
    parser.add_argument('-p', '--path', default='/tmp')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    args = parser.parse_args()
    
    with util.logging.Logger(disp_stdout=args.debug):
        if args.max_year_diff < 0:
            args.max_year_diff = float('inf')
        
        measurements.all.pw.plot.correlation_matrix_sparse_pattern(args.min_values, max_year_diff=args.max_year_diff, path=args.path)
        