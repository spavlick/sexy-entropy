from openmc.statepoint import StatePoint
import numpy as np
from uLSIF import *
from os import listdir
from os.path import isfile, join

path='statepoint-data/1D/'
files = [ f for f in listdir(path) if isfile(join(path,f)) and f.startswith('statepoint') ]

#loop through all files
for i in range(1,len(files)-1):
    sp_de=StatePoint(path+files[i-1])
    sp_nu=StatePoint(path+files[i])
    x_de=[e[1][0] for e in sp_de.source]
    x_nu=[e[1][0] for e in sp_nu.source]
    wh_x_de,wh_x_nu,wh_x_disp=uLSIF(x_de,x_nu,x_disp,fold=5,b=100)
