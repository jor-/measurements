import logging

from measurements.po4.wod.mean.estimation import save_means_from_measurements

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

save_means_from_measurements()

if print_ouput:
    print('Mean estimation saved.')