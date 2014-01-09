import logging

from measurements.po4.wod.data.io import save_measurement_dict_unsorted

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

save_measurement_dict_unsorted()

if print_ouput:
    print('Measurement dict unsorted saved.')