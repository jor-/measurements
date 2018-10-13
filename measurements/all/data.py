import measurements.dop.data
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
                     water_lsm='TMM'):
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

    # prepate water lsm
    if water_lsm is not None:
        water_lsm = water_lsm.upper()

        if water_lsm == 'TMM':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskTMM()
        elif water_lsm == 'WOA13':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13()
        elif water_lsm == 'WOA13R':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R()
        else:
            raise ValueError('Unknown land-sea-mask type {}.'.format(water_lsm))

    # create measurements collection
    measurements_collection = []

    for i, tracer in enumerate(tracers):
        # create mesurement object
        tracer = tracer.lower()
        if tracer == 'dop':
            measurements_object = measurements.dop.data.Measurements()
        elif tracer == 'po4':
            measurements_object = measurements.wod.po4.data.Measurements()
        elif tracer in ('phytoplankton', 'zooplankton'):
            measurements_object = measurements.wod.plankton.data.Measurements(tracer)
        elif tracer == 'detritus':
            measurements_object = None
        else:
            raise ValueError('Unkown tracer {}.'.format(tracer))

        # add to collection
        if measurements_object is not None and measurements_object.number_of_measurements > 0:
            # set parameters
            if water_lsm is not None:
                measurements_object.water_lsm = water_lsm
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
