#!/bin/env python

#SBATCH --job-name=full_ext_01
#SBATCH --output=ext_01_%j.out
#SBATCH --time=24:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --mem=0

#
# last edited by Claire Valva on March 17, 2019, with comments (some) added on June 24, 2019
# File uses enso spectra for a full sim



# import packages
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import scipy.integrate as sciint
import pandas as pd
from math import pi
from sympy import solve, Poly, Eq, Function, exp, re, im
from scipy.optimize import fsolve
from decimal import Decimal
import pickle
import time
import random
import multiprocessing as mp
from joblib import Parallel, delayed

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
import random

#flatten season for plotting
flatten = lambda l: [item for sublist in l for item in sublist]


# get all files

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('/scratch/midway2/clairev/detrend/spectra/'):
    f.extend(filenames)
    break

    
    
    
    # use these
def solve_f(X, Zofkt):
    # function to solve f coeff equation for trend analysis
    x,y = X
    f = Zofkt - x*np.exp(1j*y)
    return [np.real(f), np.imag(f)] 

def real_f(X,Zofkt):
    # function to wrap solve_f so that it can be used with fsolve
    x,y = X
    z = [x+0j,y+0j]
    actual_f = solve_f(z, Zofkt)
    return(actual_f)

def fwithZ(entry):
    answers = fsolve(real_f, np.array([0,0]), args = entry)
    return answers

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
    
    
# make sure this works in python 3/use python 3 for file transfer
for named in f:   
    file_name = "/scratch/midway2/clairev/detrend/" + str(named)
    file_pickle = open(file_name, "rb")
    d2_touse, d2_seasons, d2_averages = pickle.load(file_pickle)

    # i assume error might be taking the amps wrong? do after averages?


    # sort them into each season
    phase_all = [[[[fwithZ(entry) for entry in sublist] 
                   for sublist in year] 
                  for year in season] 
                 for season in d2_seasons]


    # sort them into each season
    amps_all = [[[[entry[0] for entry in sublist] 
                  for sublist in year] 
                 for year in season] 
                for season in phase_all]


    # adjust for winter averaging
    # TO DO: come up with better procedure rather 
    # current: chopping off edges to make the same length for averaging
    norml = 359
    longl = 364

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
                        for entry in amps_all[i]] 
                       for i in range(4)]


    


    #get average amplitude and phases for each season
    avg_amps = [np.average(season, axis = 0) 
                for season in season_amps_adj]

    #get average amplitude and phases for each season
    std_amps = [np.std(season, axis = 0) 
                for season in season_amps_adj]



    # get function to generate random coeffs
    def entry_fft(amp,std, phase = random.uniform(0, 2*pi)):
        # takes amplitude and phase to give corresponding fourier coeff
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
                               std = stds[wave][timed], 
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




    # set lims
    runlen = 75
    runtimes = 1
    repeattimes = 20

    listed_parts = []


    def repeater_2(amps,stds, length, times):
        #do procedure
        repeated_comp = [repeater(amps[i],stds[i], length, times)
                     for i in range(4)]
    
    #output.put(repeated_comp)
    
    
    #listed_parts.append(repeated_comp)
    
        import pickle
        
    
        file_name2 = "/scratch/midway2/clairev/from_home/sims3/01_" + str(named[10:17]) + str(random.randint(1,1000))
        file_pickle = open(file_name2,'wb') 
        pickle.dump(repeated_comp,file_pickle)
        file_pickle.close()
    
        return repeated_comp

toplot = repeater_2(avg_amps,std_amps, runlen, runtimes)
