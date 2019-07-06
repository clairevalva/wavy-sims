#!/bin/env python

#SBATCH --job-name=enso_plot
#SBATCH --output=eplt_%j.out
#SBATCH --time=24:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --mem=0


# last edited on May 23, 2019, with some comments added on June 24, 2019
# file plots the enso simulations

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


# get file names and collect sims
from os import walk
import pickle

flabs = []
for (dirpath, dirnames, filenames) in walk('gphfiles/'):
    flabs.extend(filenames)
    break

f = []
for (dirpath, dirnames, filenames) in walk('scratch-midway2/enso_sims/'):
    f.extend(filenames)
    break
    
seasons = ["winter", "spring", "summer", "fall"]
ens = ["nino", "nina", "neutral"]
d2_names = [enso + " " + part for part in seasons for enso in ens]

#make lists of el nino/regular/la nina years
nino = [1980,1983,1987,1988,1992,
        1995,1998,2003,2007,2010]
neutral = [1979,1981,1982,1984,1985,1986,1990,
           1991,1993,1994,1996,1997,2001,2002,
           2004,2005,2006,2009,2013,2014,2015,2016]
nina = [1989,1999,2000,2008,2011,2012]

len_all = 38.0
nina_per = len(nina)/len_all
nino_per = len(nino)/len_all
neutral_per = len(neutral)/len_all
all_pers = [nina_per, nino_per, neutral_per]

for wantfile in range(len(flabs)):
    
    index = str(flabs[wantfile][-10:-5])
    
    # get detrend 
    ring = 'detrended/new_detrend_' + str(index) + '.h5'
    
    data_store = pd.HDFStore(ring)
    
    # Retrieve data using key
    untrend_df = data_store['untrend_geopot']
    data_store.close()

    seasons = ["winter","spring","summer","fall"]
    # write flatten function
    
    untrend_df["seasonmean"] = untrend_df.groupby(by=['year','season'])['adj_z'].transform('mean')
    untrend_df["diff_mean"] = untrend_df["adj_z"] - untrend_df["seasonmean"]

    
    
    # go through sims to get the correct ones
    sims = []
    for name in f:
        if name[8:13] == flabs[wantfile][-10:-5]:
            file_pickle = open("scratch-midway2/enso_sims/" + name, "rb")
            sims1 = pickle.load(file_pickle)
            
            sims.append(sims1)
     
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    #flatten each 
    flat_sims = [[flatten(entry) for entry in sublist] for sublist in sims]
    
    for j in range(4):
        plt.clf();
        plt.figure(figsize=(15, 5));
        plt.hist(x = np.real(untrend_df[untrend_df["season"] == seasons[j]]["diff_mean"]), 
                     bins = 20, density = True, 
                     alpha = 0.5, label = "reanalysis")

        
        flat_all = []
        for k in range(len(flat_sims)):
            flat_tested = flat_sims[k]
            flat_all.append(flat_tested[j])
        
        
       
        flat_all = flatten(flat_all)
        for k in range(3):
        #print("hi")
            plt.hist(x = np.real(flat_all[j*3 + k]), bins = 100, 
                     density = True, alpha = 0.5, label = d2_names[j*3 + k])
        
        plt.ylabel("density")
        plt.legend()
        plt.xlabel("departure from mean geopotential height")
        plt.title(str(flabs[wantfile][-10:-5]) + " season: " +str(seasons[j]))
    
        # formatting
        sns.set_style("ticks")
        sns.set_context("poster")
        sns.despine() 
    
        plt.savefig("distributions/01_enso_plt_" + flabs[wantfile][-10:-5] + "_" +str(seasons[j]) + ".png", pad_inches=1)
        plt.close()    
        




