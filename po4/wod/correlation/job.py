import os.path
import numpy as np

import measurements.po4.wod.correlation.estimation

import util.rzcluster.interact
from util.rzcluster.job import Job


class Correlogram_Job(Job):
    
    def __init__(self, direction_index, discard_year=False, force_load=False):
        from .constants import CORRELOGRAM_JOB_OUTPUT_DIRNAME_PREFIX, CORRELOGRAM_JOB_DIRECTION_FILENAME, CORRELOGRAM_JOB_CORRELOGRAM_FILENAME
        from ..constants import MEASUREMENTS_FILE_COORDINATES
        
        ## chose values
        base_dir = measurements.po4.wod.correlation.estimation.get_base_dir(discard_year)
        
        directions = []
        if discard_year:
            from .constants import SEPARATION_VALUES_DISCARD_YEAR as separation_values
            from .constants import SAME_BOUNDS_DISCARD_YEAR as same_bounds
            from .constants import DIM_RANGES_DISCARD_YEAR as dim_ranges
            
            (t, x, y, z) = (1/365, 0.5, 0.5, 20)
            directions.extend([(t,0,0,0), (0,x,0,0), (0,0,y,0), (0,0,0,z), (t,x,0,0), (t,0,y,0), (t,0,0,z), (0,x,y,0), (0,x,0,z), (0,0,y,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
        else:
            from .constants import SEPARATION_VALUES as separation_values
            from .constants import SAME_BOUNDS as same_bounds
            from .constants import DIM_RANGES as dim_ranges
            
            (t, x, y, z) = (1/52.0, 0.5, 0.5, 20)
            directions.extend([(t,0,0,0), (0,x,0,0), (0,0,y,0), (0,0,0,z), (t,x,0,0), (t,0,y,0), (t,0,0,z), (0,x,y,0), (0,x,0,z), (0,0,y,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
            (t, x, y, z) = (1/52.0, 0.5, 0.5, 50)
            directions.extend([(t,0,0,z), (0,x,0,z), (0,0,y,z), (t,x,0,z), (t,0,y,z), (0,x,y,z), (t,x,y,z)])
            (t, x, y, z) = (1, 0.5, 0.5, 20)
            directions.extend([(t,x,0,0), (t,0,y,0), (t,0,0,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
            (t, x, y, z) = (1, 0.5, 0.5, 50)
            directions.extend([(t,0,0,z), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
            (t, x, y, z) = (1/12.0, 0.5, 0.5, 20)
            directions.extend([(t,x,0,0), (t,0,y,0), (t,0,0,z), (t,x,y,0), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
            (t, x, y, z) = (1/12.0, 0.5, 0.5, 50)
            directions.extend([(t,0,0,z), (t,x,0,z), (t,0,y,z), (t,x,y,z)])
        
        measurement_file = MEASUREMENTS_FILE_COORDINATES
        job_name = 'Corg_' + str(int(discard_year)) + '_'  + str(direction_index)
        
        
        direction = directions[direction_index]
        
        
        ## init Job
        output_dir = os.path.join(base_dir, CORRELOGRAM_JOB_OUTPUT_DIRNAME_PREFIX + str(direction_index).zfill(2))
        Job.__init__(self, output_dir, force_load=force_load)
        Job.init(self, 'westmere', 1, 1, 8, 'medium', job_name)
        
        
        ## save options
        opt = self.options
        opt['/correlogram/discard_year'] = discard_year
        opt['/correlogram/measurement_file'] = measurement_file
        opt['/correlogram/separation_values'] = separation_values
        opt['/correlogram/same_bounds'] = same_bounds
        opt['/correlogram/dim_ranges'] = dim_ranges
        opt['/correlogram/direction'] = direction
        opt['/correlogram/direction_txt_file'] = os.path.join(output_dir, 'direction.txt')
        opt['/correlogram/direction_npy_file'] = os.path.join(output_dir, CORRELOGRAM_JOB_DIRECTION_FILENAME)
        opt['/correlogram/correlogram_file'] = os.path.join(output_dir, CORRELOGRAM_JOB_CORRELOGRAM_FILENAME)
        opt['/correlogram/debug'] = True
        
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
        f.write('sys.path.append("/work_j2/sunip229/NDOP/python") \n\n')
        
        if opt['/correlogram/debug']:
            f.write('import logging \n')
            f.write('logging.basicConfig(level=logging.DEBUG) \n')
        
        f.write('from measurements.po4.wod.data.measurements import Measurements as M \n')
        f.write('m = M() \n')
        f.write('m.load("%s") \n' % opt['/correlogram/measurement_file'])
        
        if discard_year:
            f.write('m.discard_year() \n')
            f.write('m.categorize_indices(%s, t_range=%s, x_range=%s) \n' % (opt['/correlogram/separation_values'], dim_ranges[0], dim_ranges[1]))
        else:
            f.write('m.categorize_indices(%s, x_range=%s) \n' % (opt['/correlogram/separation_values'], dim_ranges[1]))
        
        f.write('m.correlogram(%s, %s, %s, wrap_around_t=%s, file="%s") \n' % (opt['/correlogram/direction'], opt['/correlogram/same_bounds'], opt['/correlogram/dim_ranges'], discard_year, opt['/correlogram/correlogram_file']))
        
        f.close()
        
        
        
        ## write job file
        f = open(opt['/job/option_file'], mode='w')
        
        f.write('#!/bin/bash \n\n')
        
        f.write('#PBS -N %s \n' % opt['/job/name'])
        f.write('#PBS -j oe \n')
        f.write('#PBS -o %s \n' % opt['/job/output_file'])
        
        try:
            f.write('#PBS -l walltime=%02i:00:00 \n' % opt['/job/walltime_hours'])
        except KeyError:
            pass
        
        f.write('#PBS -l select=%i:%s=true:ncpus=%i:mem=%igb \n' % (opt['/job/nodes'], opt['/job/cpu_kind'], opt['/job/cpus'], opt['/job/memory_gb']))
        f.write('#PBS -q %s \n\n' % opt['/job/queue'])
        
        f.write('. /usr/local/Modules/3.2.6/init/bash \n\n')
        
        f.write('module load python3 \n')
        f.write('module list \n\n')
        
        f.write('cd $PBS_O_WORKDIR \n\n')
        
        f.write('python3 %s \n' % opt['/job/execution_file'])
        f.write('touch %s \n' % opt['/job/finished_file'])
        f.write('chmod 400 %s/* \n\n' % opt['/job/output_dir'])
        
        f.write('qstat -f $PBS_JOBID \n')
        f.write('exit \n')
        
        f.close()



class Correlation_Job(Job):
    
    def __init__(self, t_factor, force_load=False):
        from .constants import SEPARATION_VALUES, SAME_BOUNDS, DIM_RANGES, CORRELATION_JOB_DIRECTION, CORRELATION_JOB_OUTPUT_DIRNAME_PREFIX, CORRELATION_JOB_MIN_MEASUREMENTS, CORRELATION_JOB_CORRELATION_FILENAME
        from ..constants import MEASUREMENTS_FILE_COORDINATES
        
        ## chose values
        base_dir = measurements.po4.wod.correlation.estimation.get_base_dir(discard_year=False)
        output_dir = os.path.join(base_dir, CORRELATION_JOB_OUTPUT_DIRNAME_PREFIX + str(t_factor).zfill(4))
        correlation_file = os.path.join(output_dir, CORRELATION_JOB_CORRELATION_FILENAME)
        job_name = 'Cor_' + str(t_factor)
        
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
        Job.__init__(self, output_dir, force_load=force_load)
        Job.init(self, 'westmere', 1, 1, 8, 'medium', job_name)
        
        
        ## save options
        opt = self.options
        opt['/correlation/measurement_file'] = MEASUREMENTS_FILE_COORDINATES
        opt['/correlation/separation_values'] = SEPARATION_VALUES
        opt['/correlation/same_bounds'] = SAME_BOUNDS
        opt['/correlation/direction'] = CORRELATION_JOB_DIRECTION
        opt['/correlation/min_measurements'] = CORRELATION_JOB_MIN_MEASUREMENTS
        opt['/correlation/wrap_around_x'] = wrap_around_ranges[1]
        opt['/correlation/t_factor'] = t_factor
        opt['/correlation/correlation_file'] = correlation_file
        opt['/correlation/debug'] = True
        
        
        
        ## write execution file
        opt['/job/execution_file'] = os.path.join(output_dir, 'run.py')
        
        f = open(opt['/job/execution_file'], mode='w')

        f.write('#!/usr/bin/env python3 \n\n')
        
        f.write('import numpy as np \n')
        f.write('import sys \n')
        f.write('sys.path.append("/work_j2/sunip229/NDOP/python") \n\n')
        
        if opt['/correlation/debug']:
            f.write('import logging \n')
            f.write('logging.basicConfig(level=logging.DEBUG) \n')
        
        f.write('from measurements.po4.wod.data.measurements import Measurements as M \n')
        f.write('m = M() \n')
        f.write('m.load("%s") \n' % opt['/correlation/measurement_file'])
        
        f.write('m.categorize_indices(%s, x_range=%s) \n' % (opt['/correlation/separation_values'], DIM_RANGES[1]))
        
        f.write('correlation = m.correlation(%s, %s, %s, %s, minimum_measurements="%d") \n' % (CORRELATION_JOB_DIRECTION, factor_list, SAME_BOUNDS, wrap_around_ranges, CORRELATION_JOB_MIN_MEASUREMENTS))
        
        f.write('np.save("%s", correlation) \n' % opt['/correlation/correlation_file'])
        
        f.close()
        
        
        
        ## write job file
        f = open(opt['/job/option_file'], mode='w')
        
        f.write('#!/bin/bash \n\n')
        
        f.write('#PBS -N %s \n' % opt['/job/name'])
        f.write('#PBS -j oe \n')
        f.write('#PBS -o %s \n' % opt['/job/output_file'])
        
        try:
            f.write('#PBS -l walltime=%02i:00:00 \n' % opt['/job/walltime_hours'])
        except KeyError:
            pass
        
        f.write('#PBS -l select=%i:%s=true:ncpus=%i:mem=%igb \n' % (opt['/job/nodes'], opt['/job/cpu_kind'], opt['/job/cpus'], opt['/job/memory_gb']))
        f.write('#PBS -q %s \n\n' % opt['/job/queue'])
        
        f.write('. /usr/local/Modules/3.2.6/init/bash \n\n')
        
        f.write('module load python3 \n')
        f.write('module list \n\n')
        
        f.write('cd $PBS_O_WORKDIR \n\n')
        
        f.write('python3 %s \n' % opt['/job/execution_file'])
        f.write('touch %s \n' % opt['/job/finished_file'])
        f.write('chmod 400 %s/* \n\n' % opt['/job/output_dir'])
        
        f.write('qstat -f $PBS_JOBID \n')
        f.write('exit \n')
        
        f.close()
