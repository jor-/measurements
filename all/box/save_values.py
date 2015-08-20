import os.path

import measurements.all.box.data as data
from measurements.all.box.constants import WOA_BASE_DIR

import measurements.all.box.constants as all
import measurements.dop.box.constants as dop
import measurements.po4.woa.data13.constants as po4

import util.logging
import util.io.fs

log_file = os.path.join(WOA_BASE_DIR, 'save_values_' + str(po4.VARI_INTERPOLATION_AMOUNT_OF_WRAP_AROUND) + '_' + str(po4.VARI_INTERPOLATION_NUMBER_OF_LINEAR_INTERPOLATOR) + '_' + str(po4.VARI_INTERPOLATION_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR) + '.log')

with util.logging.Logger(log_file=log_file, disp_stdout=not __name__ == "__main__"):
    for file in (all.NOBS_FILE, all.VARIS_FILE, all.MEANS_FILE, po4.NOBS_FILE, po4.VARIS_FILE, po4.MEANS_FILE, dop.NOBS_FILE, dop.VARIS_FILE, dop.MEANS_FILE):
        util.io.fs.remove_file(file, not_exist_okay=True)

    nobs = data.nobs()
    means = data.means()
    varis = data.variances()

