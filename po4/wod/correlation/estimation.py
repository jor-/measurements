import os.path
import numpy as np

import util.io


def get_base_dir(discard_year=False):
    from ..constants import ANALYSIS_DIR
    
    if discard_year:
        base_dir = os.path.join(ANALYSIS_DIR, 'discard_year')
    else:
        base_dir = ANALYSIS_DIR
    
    return base_dir


def get_direction_indices(discard_year=False):
    from .constants import CORRELOGRAM_DIRNAME
    
    base_dir = os.path.join(get_base_dir(discard_year), CORRELOGRAM_DIRNAME)
    number_of_directions = len(util.io.get_dirs(base_dir, with_links=False))
    direction_indices = np.arange(number_of_directions)
    
    return direction_indices


def get_output_dir(index, discard_year=False):
    from .constants import CORRELOGRAM_DIRNAME, CORRELOGRAM_JOB_OUTPUT_DIRNAME_PREFIX
    
    base_dir = get_base_dir(discard_year)
    output_dir = os.path.join(base_dir, CORRELOGRAM_DIRNAME, CORRELOGRAM_JOB_OUTPUT_DIRNAME_PREFIX + str(index).zfill(2))
    
    return output_dir



def get_direction(index, discard_year=False):
    from .constants import CORRELOGRAM_JOB_DIRECTION_FILENAME
    
    output_dir = get_output_dir(index, discard_year)
    direction_file = os.path.join(output_dir, CORRELOGRAM_JOB_DIRECTION_FILENAME)
    direction = np.load(direction_file)
    
    return direction


def get_correlogram(index, discard_year=False, axis=None, min_measurements=0):
    from .constants import CORRELOGRAM_JOB_CORRELOGRAM_FILENAME
    
    output_dir = get_output_dir(index, discard_year)
    correlogram_file = os.path.join(output_dir, CORRELOGRAM_JOB_CORRELOGRAM_FILENAME)
    correlogram = np.load(correlogram_file)
    
    if min_measurements > 0 :
        numbers = correlogram[:, 1]
        mask = numbers >= min_measurements
        correlogram = correlogram[mask]
    
    if axis is None:
        return correlogram
    else:
        return correlogram[:, axis]


def get_shift(index, discard_year=False, min_measurements=0):
    numbers = get_number(index, discard_year)
    shifts = np.arange(len(numbers))
    
    if min_measurements > 0 :
        mask = numbers >= min_measurements
        shifts = shifts[mask]
    
    return shifts



def get_function_values(function, index=None, discard_year=False):
    if index==None:
        direction_indices = get_direction_indices(discard_year)
        
        if len(direction_indices) == 0:
            raise Exception('No directions in: %s.' % get_base_dir(discard_year))
        
        values = function(direction_indices[0])
        for i in range(1, len(direction_indices)):
            direction_index = direction_indices[i]
            values = np.concatenate((values, function(direction_index)))
    else:
        values = function(index)
    
    return values

def get_correlation(index=None, discard_year=False, min_measurements=0):
    function = lambda index: get_correlogram(index, discard_year, 0, min_measurements)
    correlation = get_function_values(function, index, discard_year)
    return correlation


def get_number(index=None, discard_year=False, min_measurements=0):
    function = lambda index: get_correlogram(index, discard_year, 1, min_measurements)
    number = get_function_values(function, index, discard_year)
    return number


def get_shifted_directions(index=None, discard_year=False, min_measurements=0):
    def function(index):
        direction = get_direction(index, discard_year)
        shifts = get_shift(index, discard_year, min_measurements)
        shifted_directions = np.outer(shifts, direction)
        return shifted_directions
    
    shifted_directions = get_function_values(function, index, discard_year)
    return shifted_directions
    
    