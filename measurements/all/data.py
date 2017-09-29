import measurements.dop.data
import measurements.wod.plankton.data
import measurements.wod.po4.data
import measurements.land_sea_mask.lsm
import measurements.universal.data


TRACERS = ('dop', 'po4', 'phytoplankton', 'zooplankton')


def all_measurements(tracers=None, min_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None, near_water_lsm='TMM'):
    # check and prepare tracers
    if tracers is None:
        tracers = TRACERS
    n = len(tracers)

    # check and prepare min_standard_deviations and min_measurements_correlations
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

    min_standard_deviations = prepare_value_list(min_standard_deviations, 'min_standard_deviations')
    min_measurements_correlations = prepare_value_list(min_measurements_correlations, 'min_measurements_correlations')

    # check and prepare max_box_distance_to_water and near_water_lsm
    measurements_kargs = {}
    if max_box_distance_to_water is not None and max_box_distance_to_water != float('inf'):
        if near_water_lsm is None:
            near_water_lsm = 'TMM'
        else:
            near_water_lsm = near_water_lsm.upper()

        if near_water_lsm == 'TMM':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskTMM()
        elif near_water_lsm == 'WOA13':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13()
        elif near_water_lsm == 'WOA13R':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R()
        else:
            raise ValueError('Unknown land-sea-mask type {}.'.format(near_water_lsm))

        measurements_kargs['water_lsm'] = water_lsm
        measurements_kargs['max_box_distance_to_water'] = max_box_distance_to_water

    # create measurements collection
    measurements_collection = []

    for i, tracer in enumerate(tracers):
        tracer = tracer.lower()
        measurements_kargs['min_standard_deviation'] = min_standard_deviations[i]
        measurements_kargs['min_measurements_correlation'] = min_measurements_correlations[i]
        if tracer == 'dop':
            measurements_object = measurements.dop.data.Measurements(**measurements_kargs)
        elif tracer == 'po4':
            measurements_object = measurements.wod.po4.data.Measurements(**measurements_kargs)
        elif tracer in ('phytoplankton', 'zooplankton'):
            measurements_object = measurements.wod.plankton.data.Measurements(tracer, **measurements_kargs)
        else:
            raise ValueError('Unkown tracer {}.'.format(tracer))
        measurements_collection.append(measurements_object)

    if len(measurements_collection) > 1:
        return measurements.universal.data.MeasurementsCollectionCache(*measurements_collection)
    else:
        assert len(measurements_collection) == 1
        return measurements_collection[0]
