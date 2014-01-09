import logging

from measurements.po4.wod.data.io import save_cruises

print_ouput =  __name__ == "__main__"
if print_ouput:
    logging.basicConfig(level=logging.DEBUG)

save_cruises()

if print_ouput:
    print('Cruise list saved.')