import os.path

import numpy as np

import util.plot.save
import measurements.universal.dict
import measurements.plot.util


def histogram_and_density_function(measurements_object, file=None, min_number_of_values=100, histogram_step_size=0.05, overwrite=False, **kwargs):
    md = measurements.universal.dict.MeasurementsDict()
    md.append_values(measurements_object.points, measurements_object.values)
    lsm = measurements_object.sample_lsm
    md.categorize_indices_to_lsm(lsm, discard_year=False)
    md.filter_min_number_of_values(min_number_of_values)

    kind = 'data'
    kind_id = os.path.join(str(measurements_object.sample_lsm), 'histogram_and_density_function', f'step_size_{histogram_step_size}', f'min_values_{min_number_of_values}')
    plot_name = 'histogram_and_density_{point}'
    file = measurements.plot.util.filename(measurements_object, kind, kind_id, plot_name)

    def plot_hist(point, values):
        x_min = np.floor(min(values) * 2) / 2
        x_max = np.ceil(max(values) * 2) / 2
        point_str = ','.join(map(str, point))
        util.plot.save.histogram(file.format(point=point_str), values, step_size=histogram_step_size, x_min=x_min, x_max=x_max, density=True, add_kde=True, overwrite=overwrite, **kwargs)

    md.iterate_items(plot_hist)
