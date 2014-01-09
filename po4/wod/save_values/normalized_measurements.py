import logging

from measurements.po4.wod.correlation.io import save_normalized_measurement_dict

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

save_normalized_measurement_dict()

if print_ouput:
    print('Normalized measurement dict saved.')