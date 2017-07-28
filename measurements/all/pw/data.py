import measurements.dop.pw.data
import measurements.po4.wod.data
import measurements.universal.data
import measurements.universal.constants
import measurements.land_sea_mask.lsm


def all_measurements(max_box_distance_to_water=None, near_water_lsm='TMM', min_standard_deviation=None, min_measurements_correlation=None, tracers=None):
    if min_measurements_correlation is None:
        min_measurements_correlation = measurements.universal.constants.CORRELATION_MIN_MEASUREMENTS
    if min_standard_deviation is None:
        min_standard_deviation = measurements.constants.DEVIATION_MIN_VALUE

    measurements_collection = []

    if max_box_distance_to_water is None or max_box_distance_to_water == float('inf'):
        if tracers is None or 'po4' in tracers:
            measurements_collection.append(measurements.po4.wod.data.Measurements(min_standard_deviation=min_standard_deviation, min_measurements_correlation=min_measurements_correlation))
        if tracers is None or 'dop' in tracers:
            measurements_collection.append(measurements.dop.pw.data.Measurements(min_standard_deviation=min_standard_deviation, min_measurements_correlation=min_measurements_correlation))
    else:
        if near_water_lsm is None or near_water_lsm == 'TMM':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskTMM()
        elif near_water_lsm == 'WOA13':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13()
        elif near_water_lsm == 'WOA13R':
            water_lsm = measurements.land_sea_mask.lsm.LandSeaMaskWOA13R()
        else:
            raise ValueError('Unknown land-sea-mask type {}.'.format(near_water_lsm))

        if tracers is None or 'po4' in tracers:
            measurements_collection.append(measurements.po4.wod.data.MeasurementsNearWater(min_standard_deviation=min_standard_deviation, min_measurements_correlation=min_measurements_correlation, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water))
        if tracers is None or 'dop' in tracers:
            measurements_collection.append(measurements.dop.pw.data.MeasurementsNearWater(min_standard_deviation=min_standard_deviation, min_measurements_correlation=min_measurements_correlation, water_lsm=water_lsm, max_box_distance_to_water=max_box_distance_to_water))

    measurements_collection = measurements.universal.data.MeasurementsCollectionCache(*measurements_collection)
    return measurements_collection
