import numpy as np
from openmc.statepoint import StatePoint

sp = StatePoint('statepoint.001.h5')
xy = [elem[0][:2] for elem in sp.source]
xy = np.array(xy)
