import argparse
import os.path

import numpy as np

import measurements.po4.wod.data.results
import measurements.util.data

import util.multi_dict
import util.io.object
import util.io.fs

import util.logging

if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser(description='Save correlation of WOD data.')
    parser.add_argument('-m', '--min', type=int)
    parser.add_argument('-y', '--max_year_diff', type=int, default=1)
    # parser.add_argument('-s', '--sorted', action='store_true', help='Store sorted.')
    parser.add_argument('-c', '--categorize', action='store_true', help='Categorize.')
    parser.add_argument('--correlation', action='store_true', help='Calculate correlation.')
    parser.add_argument('--covariance', action='store_true', help='Calculate covariance.')
    parser.add_argument('-r', '--reuse', action='store_true', help='Reuse calculated values.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    args = parser.parse_args()
    
    with util.logging.Logger(disp_stdout=args.debug):
        ## chose return type
        # if args.sorted:
        #     return_type= 'measurements_sorted'
        # else:
        #     return_type= 'measurements_unsorted'
            
        ## prepare files
        file_prefix = 'measurements'
        if args.categorize:
            file_prefix = file_prefix + '_categorized'
        # if args.sorted:
        #     file_prefix = file_prefix + '_sorted_'
        # else:
        #     file_prefix = file_prefix + '_unsorted_'
        file_prefix = file_prefix + '_-_'
        from measurements.po4.wod.constants import ANALYSIS_DIR
        same_measurements_filename = file_prefix + 'same_point_measurements_-_min_{:0>2}_measurements.ppy'.format(args.min)
        same_measurements_file = os.path.join(ANALYSIS_DIR, 'correlation', same_measurements_filename)
        
        
        ## calculate measurements with same points
        if args.reuse:
            try:
                ms = measurements.util.data.MeasurementsSamePoints.load(same_measurements_file)
                found = True
            except FileNotFoundError:
                found = False
        if not args.reuse or not found:
            m = measurements.po4.wod.data.results.Measurements.load()
            if args.categorize:
                from measurements.po4.wod.constants import SAMPLE_LSM
                m.categorize_indices_to_lsm(SAMPLE_LSM, discard_year=False)
                m.means(return_type='self')
                ms = m.filter_same_points_except_year(min_values=args.min)
            else:
                from measurements.po4.wod.correlation.constants import EQUAL_BOUNDS
                ms = m.filter_same_points_with_bounds(equal_bounds=EQUAL_BOUNDS, discard_year=True, only_one_per_year=True, min_values=args.min)
            ms.save(same_measurements_file)
        if found and util.io.object.protocol_version(same_measurements_file) < 4:
            ms.save(same_measurements_file)
        util.io.fs.make_read_only(same_measurements_file)
        
        
        ## calculate covariance and correlation
        value_types = []
        if args.correlation:
            value_types.append('correlation')
        if args.covariance:
            value_types.append('covariance')
        
        for value_type in value_types:
            filename_base = file_prefix + '{}_{}' + '_-_max_{:0>2}_year_diff'.format(args.max_year_diff) + '_-_min_{:0>2}_measurements'.format(args.min) +'.{}'
            file_base = os.path.join(ANALYSIS_DIR, value_type, filename_base)
            for stationary in (False, True):
                if stationary:
                    stationary_str = 'stationary'
                    load_function = measurements.util.data.MeasurementsCovarianceStationary.load
                else:
                    stationary_str = 'nonstationary'
                    load_function = measurements.util.data.MeasurementsCovariance.load
                
                ## calculate value dict
                measurements_file = file_base.format(value_type, stationary_str, 'ppy')
                if args.reuse:
                    try:
                        mc = load_function(measurements_file)
                        found = True
                    except IOError:
                        found = False
                if not args.reuse or not found:
                    mc = ms.correlation_or_covariance(value_type, min_values=args.min, stationary=stationary, max_year_diff=args.max_year_diff)
                    mc.save(measurements_file)
                if found and util.io.object.protocol_version(measurements_file) < 4:
                    mc.save(measurements_file)
                util.io.fs.make_read_only(measurements_file)
                
                ## calculate array
                array_file = file_base.format(value_type, stationary_str, 'npy')
                if args.reuse:
                    try:
                        array = np.load(array_file)
                        found = True
                    except IOError:
                        array = None
                        found = False
                if not args.reuse or not found:
                    mc.transform_values(lambda key, value: value[1])
                    array = mc.toarray()
                    np.save(array_file, array)
                util.io.fs.make_read_only(array_file)
        
        