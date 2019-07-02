#!/bin/env python

#SBATCH --job-name=full_plot
#SBATCH --output=full_plot_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --mem=0

# coding: utf-8

# In[1]:

# this file plots the original simulations
# last edited by Claire Valva on February 23, 2019



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


# In[4]:

# get file neams for detrending
from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('gphfiles/'):
    f.extend(filenames)
    break

for wantfile in range(len(f)):
    
    # Access data store
    data_store = pd.HDFStore('new_detrend_' + str(f[wantfile][-10:-5]) + '.h5')

    # Retrieve data using key
    untrend_df = data_store['untrend_geopot']
    data_store.close()
    
    # kdljfal;sd
    untrend_df["seasonmean"] = untrend_df.groupby(by=['year','lon','season'])['adj_z'].transform('mean')
    untrend_df["diff_mean"] = untrend_df["adj_z"] - untrend_df["seasonmean"]
    
    # split dataframe into pieces so can perform fft
    untgroup = untrend_df.groupby(["year","season", "lon"])   
    other_inds = untrend_df.groupby(["year","season"]).apply(lambda x: x.name) 
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
    d2_seasons = [[d2_touse[i] for i in range(len(d2_touse)) 
                   if other_inds[i][1] == part] for part in seasons]





    len_list = [[len(entry[0]) for entry in season] for season in d2_seasons]
    max_len = [max(entry) for entry in len_list]
    padded_lists = [[[padded(row, max_len[i]) for row in entry] 
                     for entry in d2_seasons[i]] for i in range(4)]

    d2_averages = [np.average(entry, axis = 0) for entry in padded_lists]

    #save spectra:
    tosave = [d2_touse, d2_seasons, d2_averages]

    file_name = "spectra_02_" + str(f[wantfile][-10:-5]) + "arr.pickle"
    file_pickle = open(file_name, "wb")
    pickle.dump(tosave, file_pickle)

    season_titles = ["Winter", "Spring", "Summer", "Fall"]

    def spec_plot(title = 0, data = d2_averages[0], levels = "no", save = False, name = None):

        #select a season for plotting
        test_dat = data
        frequencies = fftfreq(len(test_dat[1]), 0.25)
    
        #set what you wanna crop
        max_z = 40
        max_f = 1

        #crop the data, only keep the positive frequencies
        cropped = [[test_dat[i][j] for i in range(len(zonal_spacing)) 
                    if zonal_spacing[i] <= max_z and zonal_spacing[i] >= 0]
                   for j in range(len(frequencies)) 
                   if np.abs(frequencies[j]) <= max_f]

        cropf = [counted for counted in frequencies if np.abs(counted) <= max_f]# and counted != 0]
        cropz = [zonal_spacing[i] for i in range(len(zonal_spacing)) 
                    if zonal_spacing[i] <= max_z and zonal_spacing[i] >= 0]

        x = cropf
        y = cropz
        X, Y = np.meshgrid(x,y)
        X = np.flip(X,1)
        Z = np.transpose(np.abs(cropped))

        # add cyclic point for plotting purposes
        x = np.array(x)
        testZ = [fftshift(entry) for entry in Z]
        testZ = np.array(testZ)

        dataout, lonsout = add_cyclic_point(testZ,fftshift(x))
        x = lonsout
        y = y
        X, Y = np.meshgrid(x,y)
        X = np.flip(X,1)

        # set colors and levels for discrete values
        # colors_set = cubehelix_palette(10)
        colors_set = sns.cubehelix_palette(10, start=2, rot=0, dark=0, light=.95)
    
        if levels == "no":
            # set colors and levels for discrete values
            level_set_less = [np.percentile(dataout, j*10) for j in range(1,11)]
            for j in range(1,5):
                level_set_less.append(np.percentile(dataout, 90 + 2*j))
                #level_set_less = flatten(level_set_less)
            level_set_less.sort()
            levels_rec.append(level_set_less)
        
        else:
            level_set_less = levels
        
        colors_set = sns.palplot(sns.color_palette("hls", len(level_set_less)))
        colors_set = sns.cubehelix_palette(14, start=2, rot=0, dark=0, light=.95)
        colors_set = sns.color_palette("cubehelix", 14)

        # plot it
        plt.clf();
        plt.figure(figsize=(15, 5), constrained_layout=True);
        # actual plot

        CF = plt.contourf(X,Y,dataout, colors = colors_set, levels = level_set_less,)

        # set colorbars

        CBI = plt.colorbar(CF)
        ax = CBI.ax
        ax.text(-2.5,0.8,'Coefficient magnitude',rotation=90)
    
        # ax.yaxis.get_offset_text().set_position((-3, 5))

        labels = ["{:.1E}".format(Decimal(entry)) for entry in level_set_less]

        CBI.set_ticklabels(labels)
        # plot labels
        plt.xlabel(r"Frequency (day$^{-1}$)")
        plt.ylim(ymax = 25, ymin = 3)
        plt.xlim(xmax = 0.75, xmin = -0.75)
        plt.ylabel("Zonal wavenumber")
        plt.title(str(title) + " climatology of geopotential height spectra", pad = 15)

        # formatting
        sns.set_style("ticks")
        sns.set_context("poster")
        sns.despine()
    
        if save == True:
        
            plt.savefig(name, bbox_inches = "tight")

        # plt.show()
    
    levels_rec = []

    for j in range(4):
        name = "spec_plots/02_test_spectra_" + seasons[i] + "_"+ str(f[wantfile][-10:-5]) + ".png"
        title = str(f[wantfile][-10:-5]) + " "+ season_titles[i]   
        spec_plot(title , d2_averages[i], name = name, save = True)
    
    actual_levels = np.average(levels_rec, axis = 0)

    for i in range(4):
        name = "spec_plots/02_test_spectra_" + seasons[i] + "_"+ str(f[wantfile][-10:-5]) + ".png"
        title = str(f[wantfile][-10:-5]) + " "+ season_titles[i]
        spec_plot(title , d2_averages[i], levels = actual_levels, name = name, save = True)
