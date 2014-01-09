import numpy as np

import measurements.po4.woa.data
import ndop.metos3d.data
from ndop.metos3d.model import Model

import test.parameter_sets

p = test.parameter_sets.p[0]

lsm = ndop.metos3d.data.load_land_sea_mask()
nobs = measurements.po4.woa.data.nobs()
varis = measurements.po4.woa.data.varis()
means = measurements.po4.woa.data.means()
squares = measurements.po4.woa.data.squares()

m = Model()
m_f = m.f(p, years=5)

def check_if_mask_is_equal(lsm, a):
    tracer_len, t_len, x_len, y_len, z_len  = a.shape
    for tracer in range(tracer_len):
        for t in range(t_len):
            for z in range(z_len):
                a_bool = np.isnan(a[tracer, t, ..., z])
                lsm_bool = lsm <= z
                diff = np.logical_xor(lsm_bool, a_bool)
                
                if diff.any():
                    print('tracer: ' + str(tracer) + ' time: ' + str(t) + ' differences: ' + str(diff.sum()))
                    print(np.where(diff))
                
# def check_if_mask_is_equal(a, b):
#     diff = np.logical_xor(np.isnan(a), np.isnan(b))
#     any_diff = diff.any()
#     
#     if any_diff:
#         print(np.where(diff))
#     
#     okay = not any_diff
#     
#     return okay

print('NOBS')
check_if_mask_is_equal(lsm, nobs)
print('VARIS')
check_if_mask_is_equal(lsm, varis)
print('MEANS')
check_if_mask_is_equal(lsm, means)
print('SQUARES')
check_if_mask_is_equal(lsm, squares)
print('F')
check_if_mask_is_equal(lsm, m_f)