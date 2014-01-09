import logging

from measurements.po4.wod.data.io import save_measurements

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

save_measurements()

if print_ouput:
    print('Measurement results saved.')