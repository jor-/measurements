import pathlib
import warnings

import measurements.plot.constants


def filename(measurements_object, kind, kind_id, plot_name):
    file = measurements.plot.constants.PLOT_FILE_WITHOUT_FILE_EXTENSION.format(
        tracer=measurements_object.tracer,
        data_set=measurements_object.data_set_name,
        kind=kind,
        kind_id=kind_id,
        plot_name=plot_name)
    # check for bad chars
    bad_char = '.'
    replacement_char = '_'
    if bad_char in file:
        warnings.warn(f'A "{bad_char}" is in the filename {file} which should be avoided. Replacing "{bad_char}" by "{replacement_char}".')
        file = file.replace(bad_char, replacement_char)
    assert bad_char not in file
    # append file extension
    file_extension = measurements.plot.constants.PLOT_DEFAULT_FILE_EXTENSION
    assert not file_extension.startswith('.')
    file += '.' + file_extension
    path = pathlib.PurePath(file)
    # return
    return file
