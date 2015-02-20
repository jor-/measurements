import argparse
import os.path

import util.logging

from measurements.po4.metos_boxes.mean.io import save_interpolated_list as save
from measurements.po4.metos_boxes.mean.io import get_file
from measurements.po4.metos_boxes.mean.constants import METOS_BOXES_MEANS_INTERPOLATED_LIST_FILE as file


if __name__ == "__main__":
    ## get arguments
    parser = argparse.ArgumentParser(description='Saving interpolated mean list.')
    
    parser.add_argument('-t', '--time_dim', type=int, default=2880, help='Number of time steps.')
    parser.add_argument('-w', '--amount_of_wrap_around', type=float, default=0.1, help='Amount of time and x wrap around.')
    parser.add_argument('-l', '--number_of_linear_interpolators', type=int, default=1, help='Number of linear interpolators.')
    parser.add_argument('-o', '--total_overlapping_linear_interpolators', type=float, default=0.1, help='Total overlapping of linear interpolators.')
    parser.add_argument('-m', '--min_measurements', type=int, default=1, help='Number of min measurements.')
    parser.add_argument('-p', '--parallel', action='store_true', help='Interpolate in parallel.')
    
    args = vars(parser.parse_args())
    
    time_dim = args['time_dim']
    amount_of_wrap_around = args['amount_of_wrap_around']
    number_of_linear_interpolators = args['number_of_linear_interpolators']
    total_overlapping_linear_interpolators = args['total_overlapping_linear_interpolators']
    min_measurements = args['min_measurements']
    parallel = args['parallel']
    interpolator_setup = (amount_of_wrap_around, number_of_linear_interpolators, total_overlapping_linear_interpolators)
    
    ## create logger
    file = get_file(file, time_dim, min_measurements, interpolator_setup)
    file_prefix = os.path.splitext(file)[0]
    log_file = file_prefix + '.log'
    
    with util.logging.Logger(log_file=log_file, disp_stdout=True):
        save(time_dim=time_dim, min_measurements=min_measurements, interpolator_setup=interpolator_setup, parallel=parallel)
