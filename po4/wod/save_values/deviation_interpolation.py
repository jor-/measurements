import logging

from measurements.po4.wod.deviation.io import save_deviation

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

save_deviation(deviation_file='/work_j2/sunip229/tmp/measurement_deviations_interpolated_365.npy')

if print_ouput:
    print('Standard deviation interpolation for measurements saved.')