import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp

import util.spherical


def plot_variance():
    from ..constants import MEASUREMENT_POINTS_FILE, MEASUREMENT_VARIANCE_FILE
    
    points_sperical = np.load(MEASUREMENT_POINTS_FILE)[:,1:4]
    points = util.spherical.to_cartesian(points_sperical)
    values = np.load(MEASUREMENT_VARIANCE_FILE)
    
    fig = pp.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=points[:,0], ys=points[:,1], zs=points[:,2], c=values)
    pp.show()
    #pp.savefig('/tmp/plot.png')
    
    