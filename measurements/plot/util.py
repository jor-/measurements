import measurements.plot.constants


def filename(measurements_object, kind, kind_id, plot_name):
    file = measurements.plot.constants.PLOT_FILE.format(
        tracer=measurements_object.tracer,
        data_set=measurements_object.data_set_name,
        kind=kind,
        kind_id=kind_id,
        plot_name=plot_name,
        file_extension=measurements.plot.constants.PLOT_DEFAULT_FILE_EXTENSION)
    return file
