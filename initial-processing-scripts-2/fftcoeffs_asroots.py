#!/bin/env python

#SBATCH --job-name=lock_phase
#SBATCH --output=phase_lock_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=15
#SBATCH --mem=0

# file gets roots of fft coeff (as to separate into phase and amplitude)
# last edited by Claire Valva on April 21, 2019

import numpy as np
from netCDF4 import Dataset, num2date 
from scipy.signal import get_window, csd
from scipy.fftpack import fft, ifft, fftshift, fftfreq
import pandas as pd
import datetime
from math import pi
from scipy import optimize
from astropy.stats.circstats import circmean, circvar
from os import walk
from multiprocessing import Pool

#get detrended parts
f = []
for (dirpath, dirnames, filenames) in walk('detrended/'):
    f.extend(filenames)
    break

    
#use these
def solve_f(X, solution):
    #function to solve f coeff equation for trend analysis
    x,y = X
    f = x*np.exp(1j*y) - solution
    return [np.real(f), np.imag(f)] 

def real_f(X, solution):
    #function to wrap solve_f so that it can be used with fsolve
    x,y = X
    z = [x+0j,y+0j]
    actual_f = solve_f(z, solution)
    return(actual_f)

# solve for phase
def root_find(sol):
    return optimize.root(real_f, [np.real(sol), 0], args=sol).x

#create time array
    time_list = []
    for i in range(0,55520):
        time_list.append(i*6)
    tunit = "hours since 1979-01-01T00:00:00Z"
    tarray = num2date(time_list,units = tunit,calendar = 'gregorian')

def root_find2(filed):
    
    # get dataset
    data_store = pd.HDFStore('detrended/' + filed)
    untrend_df = data_store['untrend_geopot']
    data_store.close()
    
    indexing = []
    for k in range(len(tarray)):
        if tarray[k].year == 2000:
            indexing.append(k)
        
    ind_min = np.min(indexing)
    ind_max = np.max(indexing)
    
    oned_fft = untrend_df.groupby(["time"])["adj_z"].apply(fft)
    fft_arr = np.array(oned_fft)
    
    roots = [[root_find(lon) for lon in day] for day in fft_arr]
    np.save("roots/roots_" + filed + ".npy", roots)
    
    
p = multiprocessing.Pool(processes = np.min(multiprocessing.cpu_count()-1, len(f)))


    
    
for named in f:
    p.apply_async(root_find2, named)
    
    
    
       
p.close()
p.join()
    