import os.path

import measurements.all.woa.data as data
from measurements.all.woa.constants import WOA_BASE_DIR

import measurements.all.woa.constants as all
import measurements.po4.woa.data.constants as po4
import measurements.dop.woa.constants as dop

import util.logging
import util.io

logging_file = os.path.join(WOA_BASE_DIR, 'save_values_' + str(po4.VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND) + '_' + str(po4.VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR) + '_' + str(po4.VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR) + '.log')

with util.logging.Logger(logging_file=logging_file, also_stdout = not __name__ == "__main__"):
    for file in (all.NOBS_FILE, all.VARIS_FILE, all.MEANS_FILE, po4.NOBS_FILE, po4.VARIS_FILE, po4.MEANS_FILE, dop.NOBS_FILE, dop.VARIS_FILE, dop.MEANS_FILE):
        util.io.remove_file(file, not_exist_okay=True)
    
    nobs = data.nobs()
    means = data.means()
    varis = data.varis()
    
