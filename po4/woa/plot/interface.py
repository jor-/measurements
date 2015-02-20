import numpy as np

import measurements.po4.woa.data13.load
import measurements.land_sea_mask.data
# import measurements.util.map
import util.plot



def plot_sample_mean(file='/tmp/woa_po4_sample_mean.png', vmax=None, layer=None):
    data = measurements.po4.woa.data13.load.means()
    assert data.ndim == 4
    lsm = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=len(data))
    data = lsm.apply_mask(data, land_value=np.inf)
#     for t_index in range(t_dim): 
#         data[t_index] = measurements.util.map.apply_mask(lsm, data[t_index], land_value=np.inf)
    
    if layer is not None:
        data = data[:, :, :, layer]
        data = data.reshape(data.shape + (1,))
    util.plot.data(data, file, land_value=np.inf, no_data_value=np.nan, vmin=0, vmax=vmax)


def plot_sample_deviation(file='/tmp/woa_po4_sample_deviation.png', vmax=None, layer=None):
    data = measurements.po4.woa.data13.load.variances()**(1/2)
    assert data.ndim == 4
    lsm = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=len(data))
    data = lsm.apply_mask(data, land_value=np.inf)
#     for t_index in range(len(data)): 
#         data[t_index] = measurements.util.map.apply_mask(lsm, data[t_index], land_value=np.inf)
    
    if layer is not None:
        data = data[:, :, :, layer]
        data = data.reshape(data.shape + (1,))
    util.plot.data(data, file, land_value=np.inf, no_data_value=np.nan, vmin=0, vmax=vmax)


def plot_sample_nob(file='/tmp/woa_po4_sample_nob.png', vmax=None, layer=None):
    data = measurements.po4.woa.data13.load.nobs()
    assert data.ndim == 4
    lsm = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=len(data))
    data = lsm.apply_mask(data, land_value=np.inf)
    if layer is not None:
        data = data[:, :, :, layer]
        data = data.reshape(data.shape + (1,))
    util.plot.data(data, file, no_data_value=0, vmin=1, vmax=vmax, use_log_norm=True)
