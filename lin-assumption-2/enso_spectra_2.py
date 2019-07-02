#!/bin/env python

#SBATCH --job-name=enso_full_plot
#SBATCH --output=enso_plot_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --mem=0


# last edited by Claire Valva on March 20, 2019
# this file finds average spectra for season, for enso year type for later sims



# import packages
import numpy as np
from scipy.signal import get_window, csd
from scipy.signal.windows import hann, hanning, nuttall, flattop
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.integrate as sciint
import pandas as pd
import datetime
import matplotlib.cm as cm
from math import pi
import matplotlib.ticker as tck
import datetime
from sympy import solve, Poly, Eq, Function, exp, re, im
from netCDF4 import Dataset, num2date # This is to read .nc files and time array
from scipy.optimize import fsolve
from decimal import Decimal
import pickle
import multiprocessing as mp
from joblib import Parallel, delayed
import matplotlib.colors as colors
from seaborn import cubehelix_palette #for contour plot colors
import seaborn as sns
from decimal import Decimal
import numpy.ma as ma

seasons = ["winter", "spring", "summer", "fall"]
def add_cyclic_point(data, coord=None, axis=-1):
    """
    Add a cyclic point to an array and optionally a corresponding
    coordinate.

    """
    
    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                                 len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[slicer]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value

def padded(to_pad, max_len):
    length = len(to_pad)
    zeros = max_len - length
    to_pad = list(to_pad)
    for i in range(zeros):
        to_pad.append(0)
    return to_pad

# In[3]:
zonal_spacing = fftfreq(240,1.5)
zonal_spacing = 1/zonal_spacing
zonal_spacing= 360 / zonal_spacing

#make lists of el nino/regular/la nina years
nino = [1980,1983,1987,1988,1992,
        1995,1998,2003,2007,2010]
neutral = [1979,1981,1982,1984,1985,1986,1990,
           1991,1993,1994,1996,1997,2001,2002,
           2004,2005,2006,2009,2013,2014,2015,2016]
nina = [1989,1999,2000,2008,2011,2012]

def enso_sort(year):
    if year in nino:
        return "nino"
    elif year in neutral:
        return "neutral"
    elif year in nina:
        return "nina"
    else:
        return "error"
    

# In[4]:

# get file neams for detrending
from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('gphfiles/'):
    f.extend(filenames)
    break

for wantfile in range(len(f)):
    
    # Access data store
    data_store = pd.HDFStore('detrended/new_detrend_' + str(f[wantfile][-10:-5]) + '.h5')

    # Retrieve data using key
    untrend_df = data_store['untrend_geopot']
    data_store.close()
    
    # kdljfal;sd
    # sort into seasons    
    untrend_df["enso"] = untrend_df["year"].apply(lambda x: enso_sort(x))
    untrend_df["seasonmean"] = untrend_df.groupby(by=['year','season','enso'])['adj_z'].transform('mean')
    untrend_df["diff_mean"] = untrend_df["adj_z"] - untrend_df["seasonmean"]
    
    # split dataframe into pieces so can perform fft
    untgroup = untrend_df.groupby(["year","season", "lon"])   
    other_inds = untrend_df.groupby(["year","season", "enso"]).apply(lambda x: x.name) 
    z_only = untgroup["diff_mean"]
    z_names = z_only.apply(lambda x: x.name)

    # get each longitude of each group
    grouped_a = [z_only.get_group(name) for name in z_names]

    ind_match = [[group for group in z_names 
                  if group[0] == other[0] 
                  and group[1] == other[1]] 
                 for other in other_inds]

    grouped_b = [[z_only.get_group(name) for name in sublist] 
                 for sublist in ind_match]
    grouped_b = np.real(grouped_b)

    bglist = [[list(item) for item in sublist] for sublist in grouped_b]

    d2_trans = [np.fft.fft2(sublist) for sublist in bglist]

    d2_touse = d2_trans[0:len(d2_trans)-1]
    
    ens = ["nino", "nina", "neutral"]
    d2_seasons = [[d2_touse[i] for i in range(len(d2_touse)) 
                   if other_inds[i][1] == part and other_inds[i][2] == enso]
                  for part in seasons for enso in ens]

    len_list = [[len(entry[0]) for entry in season] for season in d2_seasons]
    max_len = [max(entry) for entry in len_list]
    padded_lists = [[[padded(row, max_len[i]) for row in entry] 
                     for entry in d2_seasons[i]]
                    for i in range(len(max_len))]

    d2_averages = [np.average(entry, axis = 0) for entry in padded_lists]
    newlabels = [str(part) + "_" + str(enso) for part in seasons for enso in ens]

    #save spectra:
    tosave = [d2_touse, d2_seasons, d2_averages]

    file_name = "scratch-midway2/enso_spectra/spectra_enso_02_" + str(f[wantfile][-10:-5]) + "_arr.pickle"
    file_pickle = open(file_name, "wb")
    pickle.dump(tosave, file_pickle)



    
