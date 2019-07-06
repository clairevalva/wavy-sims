#!/bin/env python

#SBATCH --job-name=sim_lock
#SBATCH --output=sim_lock_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --mem=0

# runs sims with phase locking (didn't work, because no phase locking for some reason?)
# i wonder if this would be more accurate at a lower height
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
from sympy import Poly, Eq, Function, exp, re, im
import pickle
import time
import random
import multiprocessing as mp
from joblib import Parallel, delayed


# In[2]:


#get detrended parts
f = []
for (dirpath, dirnames, filenames) in walk('detrended/'):
    f.extend(filenames)
    break


# In[3]:


# root finders
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


# In[6]:


# get function to generate random coeffs
def entry_fft(amp, phase = random.uniform(0, 2*pi)):
    # takes amplitude and phase to give corresponding fourier coeff
    entry = amp*np.exp(1j*phase)
    return entry

# write functions to make a longer ifft
def ext_row(row, n):
    ext_f = np.zeros(((len(row) - 1) * n + 1,), dtype="complex128")
    ext_f[::n] = row * n
    
    return ext_f

def ext_ifft_new(n, input_array):
    # add the zeros onto each end
    ext_f = [ext_row(entry,n) for entry in input_array]
    
    # make up for the formulat multiplying for array length
    olddim = len(input_array[5])
    newdim = len(ext_f[0])
    mult = newdim/olddim
    
    ext_f = np.multiply(ext_f, mult)
    adjusted_tested = np.fft.ifft2(ext_f)
    
    return adjusted_tested


season_titles = ["Winter", "Spring", "Summer", "Fall"]

# flatten season for plotting
flatten = lambda l: [item for sublist in l for item in sublist]


# In[23]:


file_pickle = open("spectra_copy/spectra_02_45.0Narr.pickle", "rb")
d2_touse, d2_seasons, d2_averages = pickle.load(file_pickle)


# In[9]:


filed = ["spectra/spectra_02_45.0Sarr.pickle", 
         "spectra/spectra_02_45.0Narr.pickle"]


# In[39]:


mean_phases_lock = [[-1.20929458e-16,  1.65918271e-01, -2.17292412e-01, -2.40352609e-01,
        8.64205449e-02,  1.07202695e-02],[-1.21105919e-16,  3.96836386e-01, -5.77513605e-01,  5.62200988e-01,
        3.64883992e-01, -7.35447431e-02]]


# In[40]:


stds_lock = [[0.        , 0.35092645, 0.37481109, 0.36100874, 0.35869798,
       0.36139656], [0.        , 0.22681627, 0.40726111, 0.39961749, 0.37641118,
       0.36154913]]


# In[38]:


for heythere in range(1):
# get function to generate random coeffs
    def entry_fft(amp,std,wavenum, phase = random.uniform(0, 2*pi)):
    # takes amplitude and phase to give corresponding fourier coeff
        if np.abs(wavenum) <= 6:
            phase = np.random.normal(loc = mean_phases_lock[ko][wavenum], scale = stds_lock[ko][wavenum])
        
        amp_new = np.random.normal(loc = amp, scale = std)
        entry = amp_new*np.exp(1j*phase)
        return entry
    
    # write functions to make a longer ifft
    def ext_row(row, n):
        ext_f = np.zeros(((len(row) - 1) * n + 1,), dtype="complex128")
        ext_f[::n] = row * n
    
        return ext_f

    def ext_ifft_new(n, input_array):
    # add the zeros onto each end
        ext_f = [ext_row(entry,n) for entry in input_array]
    
        # make up for the formulat multiplying for array length
        olddim = len(input_array[5])
        newdim = len(ext_f[0])
        mult = newdim/olddim
    
        # ext_f = np.multiply(mult, ext_f)
        adjusted_tested = np.fft.ifft2(ext_f)
    
        return adjusted_tested
    
    def combined(amps,stds, length):
    # combines generation of random phase with inverse transform
        newarray = [[entry_fft(amp = amps[wave][timed],
                               std = stds[wave][timed], wavenum = wave, 
                               phase = random.uniform(0, 2*pi)) 
                     for timed in range(len(amps[wave]))]
                    for wave in range(len(amps))]
    
        newarray = [np.array(leaf) for leaf in newarray]
        iffted = ext_ifft_new(length, newarray)
        
        return iffted


# In[11]:


for ko in range(2):
    file_pickle = open(filed[ko], "rb")
    d2_touse, d2_seasons, d2_averages = pickle.load(file_pickle)
    
    
    alled = [[[[root_find(entry) for entry in sublist] 
                   for sublist in year] 
                  for year in season] 
                 for season in d2_seasons]
    phases = alled[:,:,:,1]
    amps = alled[:,:,:,0]

    
    def padded(to_pad, index):
        length = len(to_pad)
        if index == 0:
            zeros = longl - length
            to_pad = list(to_pad)
            for i in range(zeros):
                to_pad.append(0)
            return to_pad
        else:
            return to_pad

    #pad rows with zeros to account for leap year
    season_amps_adj = [[[padded(row, index = i)  
                         for row in entry] 
                        for entry in amps[i]] 
                       for i in range(4)]

    #get average amplitude and phases for each season
    avg_amps = [np.average(season, axis = 0) 
                for season in season_amps_adj]

    #get average amplitude and phases for each season
    std_amps = [np.std(season, axis = 0) 
                for season in season_amps_adj]
    
    def entry_fft(amp,std,wavenum, phase = random.uniform(0, 2*pi)):
    # takes amplitude and phase to give corresponding fourier coeff
        if np.abs(wavenum) <= 6:
            phase = np.random.normal(loc = mean_phases_lock[ko][wavenum], scale = stds_lock[ko][wavenum])
        
        amp_new = np.random.normal(loc = amp, scale = std)
        entry = amp_new*np.exp(1j*phase)
        return entry
    
    # write functions to make a longer ifft
    def ext_row(row, n):
        ext_f = np.zeros(((len(row) - 1) * n + 1,), dtype="complex128")
        ext_f[::n] = row * n
    
        return ext_f

    def ext_ifft_new(n, input_array):
    # add the zeros onto each end
        ext_f = [ext_row(entry,n) for entry in input_array]
    
        # make up for the formulat multiplying for array length
        olddim = len(input_array[5])
        newdim = len(ext_f[0])
        mult = newdim/olddim
    
        # ext_f = np.multiply(mult, ext_f)
        adjusted_tested = np.fft.ifft2(ext_f)
    
        return adjusted_tested
    
    def combined(amps,stds, length):
    # combines generation of random phase with inverse transform
        newarray = [[entry_fft(amp = amps[wave][timed],
                               std = stds[wave][timed], wavenum = wave, 
                               phase = random.uniform(0, 2*pi)) 
                     for timed in range(len(amps[wave]))]
                    for wave in range(len(amps))]
    
        newarray = [np.array(leaf) for leaf in newarray]
        iffted = ext_ifft_new(length, newarray)
        
        return iffted
    
    def repeater(season, stds, length, times):
        # repeats the phase creation and inverse transform
        newarray = [combined(season,stds,length) for leaf in range(times)] 
        return(newarray)
    
    def repeater_2(amps,stds, length, times):
        #do procedure
        repeated_comp = [repeater(amps[i],stds[i], length, times)
                     for i in range(4)]
    
    #output.put(repeated_comp)
    
    
    #listed_parts.append(repeated_comp)
    
        import pickle
        
    
        file_name2 = "sim_samp/"
        file_pickle = open(file_name2,'wb') 
        pickle.dump(repeated_comp,file_pickle)
        file_pickle.close()
    
        return repeated_comp
    
    
    runlen = 70
    runtimes = 4
    toplot = repeater_2(avg_amps,std_amps, runlen, runtimes)    


# In[ ]:





# In[ ]:




