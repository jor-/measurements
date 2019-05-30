import measurements.dop.data
import measurements.wod.constants
import measurements.wod.plankton.data
import measurements.wod.po4.data
import measurements.land_sea_mask.lsm
import measurements.universal.data

import util.logging


TRACERS = ('dop', 'po4', 'phytoplankton', 'zooplankton', 'detritus')
LAND_SEA_MASKS = ('TMM', 'WOA13', 'WOA13R')


def all_measurements(tracers=None,
                     min_measurements_mean=None,
                     min_measurements_quantile=None,
                     min_measurements_standard_deviation=None,
                     min_measurements_correlation=None,
                     min_standard_deviation=None,
                     max_box_distance_to_water=None,
                     water_lsm=None,
                     sample_lsm=None):
    # check and prepare tracers
    if tracers is None:
        tracers = TRACERS
    n = len(tracers)

    # check and prepare parameters
    def prepare_value_list(values, values_name):
        if values is None:
            values = (None,) * n
        else:
            try:
                m = len(values)
            except TypeError:
                values = (values,) * n
            else:
                if m == 1:
                    values = values * n
                elif m != n:
                    raise ValueError('{values_name} must be an iterable with the same number of values as tracers ({tracers_len}) or one value but it has {values_len} values.'.format(values_name=values_name, tracers_len=n, values_len=m))
        return values

    min_standard_deviation = prepare_value_list(min_standard_deviation, 'min_standard_deviation')
    min_measurements_mean = prepare_value_list(min_measurements_mean, 'min_measurements_mean')
    min_measurements_quantile = prepare_value_list(min_measurements_quantile, 'min_measurements_quantile')
    min_measurements_standard_deviation = prepare_value_list(min_measurements_standard_deviation, 'min_measurements_standard_deviation')
    min_measurements_correlation = prepare_value_list(min_measurements_correlation, 'min_measurements_correlation')

    # prepare lsms
    def lsm_by_name(name, t_dim=None):
        if name is not None:
            name = name.upper()
            if name == 'TMM':
                return measurements.land_sea_mask.lsm.LandSeaMaskTMM(t_dim=t_dim)
            elif name == 'WOA13':
                return measurements.land_sea_mask.lsm.LandSeaMaskWOA13(t_dim=t_dim)
            elif name == 'WOA13R':
                return measurements.land_sea_mask.lsm.LandSeaMaskWOA13R(t_dim=t_dim)
            else:
                raise ValueError('Unknown land-sea-mask type {}.'.format(name))
        return None

    water_lsm = lsm_by_name(water_lsm)
    sample_lsm = lsm_by_name(sample_lsm, t_dim=12)

    # create measurements collection
    measurements_collection = []

    for i, tracer in enumerate(tracers):
        # create mesurement object
        tracer = tracer.lower()
        if tracer == 'dop':
            measurements_object = measurements.dop.data.Measurements()
        elif tracer == 'po4':
            measurements_object = measurements.wod.po4.data.Measurements(measurements.wod.constants.WOD_13_DATA_SET_NAME)
        elif tracer in ('phytoplankton', 'zooplankton'):
            measurements_object = measurements.wod.plankton.data.Measurements(tracer, measurements.wod.constants.WOD_13_DATA_SET_NAME)
        elif tracer == 'detritus':
            measurements_object = None
        else:
            raise ValueError('Unkown tracer {}.'.format(tracer))

        # add to collection
        if measurements_object is not None and measurements_object.number_of_measurements > 0:
            # set parameters
            if water_lsm is not None:
                measurements_object.water_lsm = water_lsm
            if sample_lsm is not None:
                measurements_object.sample_lsm = sample_lsm
            if max_box_distance_to_water is not None:
                measurements_object.max_box_distance_to_water = max_box_distance_to_water

            if min_measurements_mean[i] is not None:
                measurements_object.min_measurements_mean = min_measurements_mean[i]
            if min_measurements_quantile[i] is not None:
                measurements_object.min_measurements_quantile = min_measurements_quantile[i]
            if min_measurements_standard_deviation[i] is not None:
                measurements_object.min_measurements_standard_deviation = min_measurements_standard_deviation[i]
            if min_measurements_correlation[i] is not None:
                measurements_object.min_measurements_correlation = min_measurements_correlation[i]

            if min_standard_deviation[i] is not None:
                measurements_object.min_standard_deviation = min_standard_deviation[i]

            # add to collection
            util.logging.debug('Measurements {} used for tracer {} with {} data.'.format(measurements_object, tracer, measurements_object.number_of_measurements))
            measurements_collection.append(measurements_object)

    number_of_measurements_objects = len(measurements_collection)
    if number_of_measurements_objects > 1:
        return measurements.universal.data.MeasurementsCollectionCache(*measurements_collection)
    elif number_of_measurements_objects == 1:
        return measurements_collection[0]
    else:
        util.logging.warning('No measurements found for {} tracers.'.format(tracers))
        return None
