#!/bin/env python

#SBATCH --job-name=stat_full
#SBATCH --output=statplt_%j.out
#SBATCH --time=24:05:00
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --mem=0
# #SBATCH --exclusive
# this one did work

# last edited on May 12, 2019, with update and cleaning on June 24, 2019
# this file gets summary stats from original simulations
# note that there is a small error in the skew, which was fixed in csv via an approx with sd, mean, median

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
from scipy.stats import skew


# get file names and collect sims
from os import walk
import pickle

flabs = []
for (dirpath, dirnames, filenames) in walk('gphfiles/'):
    flabs.extend(filenames)
    break

f = []
for (dirpath, dirnames, filenames) in walk('/scratch/midway2/clairev/from_home/01_full_sims/'):
    f.extend(filenames)
    break
    
jjj = 0

frames = []
    
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

    winter_means = list(untrend_df[untrend_df["season"]
                                   == seasons[0]].groupby(['lon'])["adj_z"].mean())
    spring_means = list(untrend_df[untrend_df["season"] 
                                   == seasons[1]].groupby(['lon'])["adj_z"].mean())
    summer_means = list(untrend_df[untrend_df["season"] 
                                   == seasons[2]].groupby(['lon'])["adj_z"].mean())
    fall_means = list(untrend_df[untrend_df["season"]
                                 == seasons[3]].groupby(['lon'])["adj_z"].mean())

    all_means = [winter_means, spring_means, summer_means, fall_means]
    
    # go through sims to get the correct ones
    sims = []
    for name in f:
        if name[4:9] == flabs[wantfile][-10:-5]:
            file_pickle = open("/scratch/midway2/clairev/from_home/01_full_sims/" + name, "rb")
            sims1 = pickle.load(file_pickle)
            newsims = [[list(np.add(sims1[season][0][j],
                                    np.real(all_means[season][j]))) 
                        for j in range(len(sims1[season][0]))]
                       for season in range(4)]
            sims.append(newsims)
     
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    #flatten each 
    flat_sims = [[flatten(entry) for entry in sublist] for sublist in sims]
    
    sertest = (entry for entry in sims)
    teststack = np.dstack((sims[0], sims[1]))
    newstack = [[flatten(entry) for entry in season] 
                for season in teststack]
    
    # get stats
    runnumber = len(sims)
    entry_len = len(sims[0][0][0])
    
    lon_avgs = [[np.average(entry) for entry in season] for season in newstack]
    lon_vars = [[np.var(entry) for entry in season] for season in newstack]
    lon_median = [[np.median(entry) for entry in season] for season in newstack]
    lon_skew = [[skew(entry) for entry in season] for season in newstack]

    full_skew = [skew(season) for season in newstack]
    full_avg = [np.median(season) for season in newstack]
    full_var = [np.average(season) for season in newstack]
    full_median = [np.var(season) for season in newstack]
    
    # now put everythig into a dataframe
    for season in range(4):
    
        meds = lon_median[season]
        meds.append(full_median[season])
    
        skewed = lon_skew[season]
        skewed.append(full_skew[season])
    
        varr = lon_vars[season]
        varr.append(full_var[season])
    
        avgs = lon_avgs[season]
        avgs.append(full_avg[season])
    
        lon_list = [i*1.5 for i in range(240)]
        lon_list.append(500.0)
    
        runtimes = [runnumber for i in range(241)]
        entrylen = [entry_len for i in range(241)]
        version = [flabs[wantfile][-10:-5] for i in range(241)]
        seased = [seasons[season] for i in range(241)]
    
        #append all to a temp frame
        d = {"version" : version, "runtimes" : runtimes, "entrylen" : entrylen,
             "lon": lon_list, "median" : meds, "skew" : skewed,
             "variance":varr, "average" : avgs, "season": seased}
    
        df = pd.DataFrame(data=d)
        
        frames.append(df)
        

#merge to all dataframes
alldf = pd.concat(frames)

alldf.to_pickle("sims1_stats.pkl")

store = pd.HDFStore('sims1_stats.h5')
store.append('stats_1', alldf)
store.close()

    
        
    


