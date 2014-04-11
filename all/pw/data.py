import measurements.dop.pw.data
import measurements.po4.wod.data.io

def get_points_and_values():
    (dop_points, dop_values) = measurements.dop.pw.data.load_points_and_values()
    po4_points = measurements.po4.wod.data.io.load_measurement_points()
    po4_values = measurements.po4.wod.data.io.load_measurement_results()
    
    points = (dop_points, po4_points)
    values = (dop_values, po4_values)
    
    return (points, values)



import measurements.dop.pw.deviation
import measurements.po4.wod.deviation.io

def get_deviation():
    dop_deviation = measurements.dop.pw.deviation.get_deviation()
    po4_deviation = measurements.po4.wod.deviation.io.load_deviations()
    
    deviation = (dop_deviation, po4_deviation)
    
    return deviation