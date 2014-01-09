import logging

from measurements.po4.wod.deviation.estimation import save_deviations_from_measurements

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

save_deviations_from_measurements(deviations_file='/work_j2/sunip229/tmp/measurement_deviations_estimation_365.npy', separation_values= [1./365., 1, 1, [0, 25, 50, 85, 120, 170, 220, 290, 360, 455, 550, 670, 790, 935, 1080, 1250, 1420, 1615, 1810, 2030, 2250, 2495, 2740, 3010, 3280, 3575, 3870, 4190, 4510, 6755, 9000]])
# save_deviations_from_measurements(deviations_file='/work_j2/sunip229/tmp/measurement_deviations_estimation_52.npy', separation_values= [1./52., 1, 1, [0, 25, 50, 85, 120, 170, 220, 290, 360, 455, 550, 670, 790, 935, 1080, 1250, 1420, 1615, 1810, 2030, 2250, 2495, 2740, 3010, 3280, 3575, 3870, 4190, 4510, 6755, 9000]])

if print_ouput:
    print('Standard deviation estimation saved.')