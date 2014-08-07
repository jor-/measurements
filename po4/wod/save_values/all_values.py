import logging

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

#import measurements.po4.wod.save_values.cruises_list
#import measurements.po4.wod.save_values.measurement_dict_unsorted
#import measurements.po4.wod.save_values.measurement_results
#import measurements.po4.wod.save_values.deviation_estimation
import measurements.po4.wod.save_values.deviation_interpolation
import measurements.po4.wod.save_values.means_estimation
import measurements.po4.wod.save_values.measurement_dict_sorted

if print_ouput:
    print('All values saved.')