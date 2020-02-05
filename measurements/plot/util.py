import pathlib
import warnings

import numpy as np

import util.plot.auxiliary
import measurements.plot.constants


def filename(measurements_object, kind, kind_id, plot_name):
    file = measurements.plot.constants.PLOT_FILE_WITHOUT_FILE_EXTENSION.format(
        tracer=measurements_object.tracer,
        data_set=measurements_object.data_set_name,
        kind=kind,
        kind_id=kind_id,
        plot_name=plot_name)
    # replace bad chars
    file = util.plot.auxiliary.replace_bad_characters(file)
    # append file extension
    file_extension = measurements.plot.constants.PLOT_DEFAULT_FILE_EXTENSION
    assert not file_extension.startswith('.')
    file += '.' + file_extension
    path = pathlib.PurePath(file)
    # return
    return file


def append_to_filename(filename, suffix):
    path = pathlib.PurePath(filename)
    new_suffix = suffix + path.suffix
    path = path.parent.joinpath(path.stem + new_suffix)
    return path


def append_v_max_to_filename(filename, v_max):
    if v_max is not None:
        filename = append_to_filename(filename, f'_-_v_max_{v_max}')
    return filename


def change_one_dim(data, new_dim=None, axis=0):
    if new_dim is not None:
        new_dim = int(new_dim)
        old_dim = data.shape[axis]
        factor = old_dim / new_dim
        if factor.is_integer() and factor >= 1:
            if factor > 1:
                factor = int(factor)
                new_shape = data.shape[:axis] + (new_dim,) + data.shape[axis + 1:]
                new_data = np.zeros(new_shape)
                prefix = (slice(None, None, None),) * axis
                for i in range(factor):
                    new_data += data[prefix][i::factor]
                new_data /= factor
            else:
                new_data = data
        else:
            raise ValueError(f'Old dim {old_dim} must be a mutiple of new dim {new_dim}.')
        assert new_data.ndim == data.ndim
        assert new_data.shape[axis] == new_dim
        return new_data
    else:
        return data
