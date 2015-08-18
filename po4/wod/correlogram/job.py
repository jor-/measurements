import os.path
import numpy as np

import measurements.po4.wod.correlogram.estimation

import util.batch.universal.system


class Correlogram_Job(util.batch.universal.system.Job):
    
    def __init__(self, direction_index, discard_year=False, cpu_kind='f_ocean2', force_load=False, debug=False):
        from .constants import MEASUREMENTS_NORMALIZED_DICT_FILE, CORRELOGRAM_DIRNAME, CORRELOGRAM_JOB_OUTPUT_DIRNAME_PREFIX, CORRELOGRAM_JOB_DIRECTION_FILENAME, CORRELOGRAM_JOB_CORRELOGRAM_FILENAME
        from constants import PYTHON_DIR
        
        ## chose values
        base_dir = measurements.po4.wod.correlation.estimation.get_base_dir(discard_year)
        
        directions = []
        if discard_year:
            from .constants import SAME_BOUNDS_DISCARD_YEAR as same_bounds
            from .constants import DIM_RANGES_DISCARD_YEAR as dim_ranges
            
            (t, x, y, z) = (1/365, 0.5, 0.5, 20)
            directions.extend([(t,0,0,0), (0,x,0,0), (0,0,y,0), (0,0,0,z), (t,x,0,0), (t,0,y,0), (t,0,0,z), (0,x,y,0), (0,x,0,z), (0,0,y,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
        else:
            from .constants import SAME_BOUNDS as same_bounds
            from .constants import DIM_RANGES as dim_ranges
            
#             (t, x, y, z) = (1/52.0, 0.5, 0.5, 20)
#             directions.extend([(t,0,0,0), (0,x,0,0), (0,0,y,0), (0,0,0,z), (t,x,0,0), (t,0,y,0), (t,0,0,z), (0,x,y,0), (0,x,0,z), (0,0,y,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
#             (t, x, y, z) = (1/52.0, 0.5, 0.5, 50)
#             directions.extend([(t,0,0,z), (0,x,0,z), (0,0,y,z), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
#             (t, x, y, z) = (1, 0.5, 0.5, 20)
#             directions.extend([(t,x,0,0), (t,0,y,0), (t,0,0,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
#             (t, x, y, z) = (1, 0.5, 0.5, 50)
#             directions.extend([(t,0,0,z), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
#             (t, x, y, z) = (1/12.0, 0.5, 0.5, 20)
#             directions.extend([(t,x,0,0), (t,0,y,0), (t,0,0,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
#             (t, x, y, z) = (1/12.0, 0.5, 0.5, 50)
#             directions.extend([(t,0,0,z), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
            
            (t, x, y, z) = (1/365.0, 0.5, 0.5, 20)
            directions.extend([(t,0,0,0), (0,x,0,0), (0,0,y,0), (0,0,0,z), (t,x,0,0), (t,0,y,0), (t,0,0,z), (0,x,y,0), (0,x,0,z), (0,0,y,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
            (t, x, y, z) = (1/52.0, 0.5, 0.5, 20)
            directions.extend([(t,x,0,0), (t,0,y,0), (t,0,0,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
            (t, x, y, z) = (1/12.0, 0.5, 0.5, 20)
            directions.extend([(t,x,0,0), (t,0,y,0), (t,0,0,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
            
            (t, x, y, z) = (1/365.0, 0.5, 0.5, 50)
            directions.extend([(t,0,0,z), (0,x,0,z), (0,0,y,z), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
            (t, x, y, z) = (1/52.0, 0.5, 0.5, 50)
            directions.extend([(t,0,0,z), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
            (t, x, y, z) = (1/12.0, 0.5, 0.5, 50)
            directions.extend([(t,0,0,z), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
        
        job_name = 'CGram_' + str(int(discard_year)) + '_'  + str(direction_index)
        
        direction = directions[direction_index]
        
        
        ## init Job
        output_dir = os.path.join(base_dir, CORRELOGRAM_DIRNAME, CORRELOGRAM_JOB_OUTPUT_DIRNAME_PREFIX + str(direction_index).zfill(2))
        super().__init__(output_dir, force_load=force_load)
        nodes_setup = util.batch.universal.system.NodeSetup(memory=7, node_kind=cpu_kind, cpus=1, nodes=1)
        # super().init_job_file(job_name, 7, [cpu_kind, 1, 1])
        super().init_job_file(job_name, nodes_setup)
        
        
        ## save options
        opt = self.options
        opt['/correlogram/discard_year'] = discard_year
        opt['/correlogram/measurement_file'] = MEASUREMENTS_NORMALIZED_DICT_FILE
        opt['/correlogram/same_bounds'] = same_bounds
        opt['/correlogram/dim_ranges'] = dim_ranges
        opt['/correlogram/direction'] = direction
        opt['/correlogram/direction_txt_file'] = os.path.join(output_dir, 'direction.txt')
        opt['/correlogram/direction_npy_file'] = os.path.join(output_dir, CORRELOGRAM_JOB_DIRECTION_FILENAME)
        opt['/correlogram/correlogram_file'] = os.path.join(output_dir, CORRELOGRAM_JOB_CORRELOGRAM_FILENAME)
        opt['/correlogram/debug'] = debug
        opt['/correlogram/is_normalized'] = True
        
        ## save direction
        direction_array = np.array(direction)
        np.savetxt(opt['/correlogram/direction_txt_file'], direction_array)
        np.save(opt['/correlogram/direction_npy_file'], direction_array)
        
        
        ## write execution file
        opt['/job/execution_file'] = os.path.join(output_dir, 'run.py')
        
        f = open(opt['/job/execution_file'], mode='w')

        f.write('#!/usr/bin/env python3 \n\n')
        
        f.write('import numpy as np \n')
        f.write('import sys \n')
        f.write('sys.path.append("' + PYTHON_DIR + '") \n\n')
        
        if opt['/correlogram/debug']:
            f.write('import logging \n')
            f.write('logging.basicConfig(level=logging.DEBUG) \n')
        
        f.write('from measurements.po4.wod.data.results import Measurements as M \n')
        f.write('m = M() \n')
        f.write('m.load("%s") \n' % opt['/correlogram/measurement_file'])
        
        if discard_year:
            f.write('m.discard_year() \n')
        
        f.write('m.total_correlogram(%s, %s, %s, wrap_around_t=%s, is_normalized=%d, file="%s") \n' % (direction, same_bounds, dim_ranges, discard_year, True, opt['/correlogram/correlogram_file']))
        
        f.close()
        
        ## write job file
        run_command = 'python3 %s \n' % opt['/job/execution_file']
        self.write_job_file(run_command, modules=('python3',))
        
        f.close()



class Correlation_Job(Job):
    
    def __init__(self, t_factor, cpu_kind='f_ocean2', force_load=False, debug=False):
        from .constants import MEASUREMENTS_NORMALIZED_DICT_FILE, SAME_BOUNDS, DIM_RANGES, CORRELATION_JOB_DIRECTION, CORRELATION_JOB_OUTPUT_DIRNAME_PREFIX, CORRELATION_JOB_MIN_MEASUREMENTS, CORRELATION_JOB_CORRELATION_FILENAME
        from constants import PYTHON_DIR
#         from ..constants import MEASUREMENTS_FILE_COORDINATES
        
        ## chose values
        base_dir = measurements.po4.wod.correlation.estimation.get_base_dir(discard_year=False)
        output_dir = os.path.join(base_dir, CORRELATION_JOB_OUTPUT_DIRNAME_PREFIX + str(t_factor).zfill(4))
        correlation_file = os.path.join(output_dir, CORRELATION_JOB_CORRELATION_FILENAME)
        job_name = 'CLation_' + str(t_factor)
        
        wrap_around_ranges = (None, DIM_RANGES[1], None, None)
        
        direction_array = np.array(CORRELATION_JOB_DIRECTION)
        dim_ranges_array = np.array(DIM_RANGES)
        max_factors = (dim_ranges_array[:,1] - dim_ranges_array[:,0]) / direction_array
        max_factors[1] /= 2
        max_factors = np.ceil(max_factors).astype(int)
        factor_list = ((t_factor,),)
        for i in range(1,4):
            factor_list += (range(max_factors[i]),)
        
        ## init Job
        super().__init__(output_dir, force_load=force_load)
        nodes_setup = util.batch.universal.system.NodeSetup(memory=7, node_kind=cpu_kind, cpus=1, nodes=1)
        # super().init_job_file(1, 1, 7, job_name, cpu_kind=cpu_kind)
        super().init_job_file(job_name, nodes_setup)
        
        
        
        ## save options
        opt = self.options
        opt['/correlation/measurement_file'] = MEASUREMENTS_NORMALIZED_DICT_FILE
#         opt['/correlation/separation_values'] = SEPARATION_VALUES
        opt['/correlation/same_bounds'] = SAME_BOUNDS
        opt['/correlation/direction'] = CORRELATION_JOB_DIRECTION
        opt['/correlation/min_measurements'] = CORRELATION_JOB_MIN_MEASUREMENTS
        opt['/correlation/wrap_around_x'] = wrap_around_ranges[1]
        opt['/correlation/t_factor'] = t_factor
        opt['/correlation/correlation_file'] = correlation_file
        opt['/correlation/debug'] = debug
        opt['/correlation/is_normalized'] = True
        
        
        
        ## write execution file
        opt['/job/execution_file'] = os.path.join(output_dir, 'run.py')
        
        f = open(opt['/job/execution_file'], mode='w')

        f.write('#!/usr/bin/env python3 \n\n')
        
        f.write('import numpy as np \n')
        f.write('import sys \n')
        f.write('sys.path.append("' + PYTHON_DIR + '") \n\n')
        
        if opt['/correlation/debug']:
            f.write('import logging \n')
            f.write('logging.basicConfig(level=logging.DEBUG) \n')
        
        f.write('from measurements.po4.wod.data.results import Measurements as M \n')
        f.write('m = M() \n')
        f.write('m.load("%s") \n' % opt['/correlation/measurement_file'])
        
        f.write('m.total_correlation(%s, %s, %s, %s, minimum_measurements=%d, is_normalized=%d, file="%s") \n' % (CORRELATION_JOB_DIRECTION, factor_list, SAME_BOUNDS, wrap_around_ranges, CORRELATION_JOB_MIN_MEASUREMENTS, True, correlation_file))
        
        f.close()
        
        
        
        ## write job file
        run_command = 'python3 %s \n' % opt['/job/execution_file']
        self.write_job_file(run_command, modules=('python3',))
