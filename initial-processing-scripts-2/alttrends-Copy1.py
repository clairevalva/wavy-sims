#!/bin/env python

#SBATCH --job-name=full_alt_trending
#SBATCH --output=full_please_%j.out
#SBATCH --time=30:30:00
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive

# coding: utf-8

# In[1]:

# last edited by Claire Valva on February 23, 2019
# this file is a copy of original file used to remove seasonal trends from .nc files
# the detrended files are saved as a netcdf


# In[2]:

# import packages
import numpy as np
from scipy.signal import get_window, csd
from scipy.signal.windows import hann, hanning, nuttall, flattop
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
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
import pickle


# In[3]:

# In[3]:
zonal_spacing = fftfreq(240,1.5)
zonal_spacing = 1/zonal_spacing
zonal_spacing= 360 / zonal_spacing

def flatten(l):
    #flattens lists
    return [item for sublist in l for item in sublist]

# append season to each row
def season_sort(x):
    if x < 3:
        return("winter")
    elif x >= 12:
        return("winter")
    elif x >= 3 and x < 6:
        return("spring")
    elif x >= 6 and x < 9:
        return("summer")
    elif x >= 9 and x < 12:
        return("fall")
    else:
        return("error?")


# In[4]:

# get file neams for detrending
from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('gphfiles/'):
    f.extend(filenames)
    break


# In[8]:

# get lists for processing
year_number = 2016 - 1979 + 1
year_list = [1979 + i for i in range(2017-1979)]

# create time array
time_list = []
for i in range(0,55520):
    time_list.append(i*6)
tunit = "hours since 1979-01-01T00:00:00Z"
tarray = num2date(time_list,units = tunit,calendar = 'gregorian')

# create longitude array
lon_increment = 1.5 # The increment of each longitude grid is 1.5
lon_list = [i * lon_increment for i in range(240)]


# In[9]:

# get fft results at each date over entire longitude
wavenum_zone = fftfreq(372, 1.5)
wavenum_zone = 1/wavenum_zone
wavenum_zone = 372*1.5 / wavenum_zone

# get list of years and seasons to perform transform/detrend on
years = range(1979,2017)
seasons = ["winter", "spring", "summer", "fall"]


# In[176]:

for wantfile in range(len(f)):


    #get file

    filepath = 'gphfiles/' + f[wantfile]
    fileobj = Dataset(filepath, mode='r')


    # set indicies/number of things
    number_entries = int(fileobj.dimensions['time'].size)
    number_days = int(number_entries / 4)
    number_lon = fileobj.dimensions['longitude'].size
        
    # get indexing list
    lon_list_df = [lon_list[k] 
                for i in range(number_entries)
                for k in range(number_lon)]

    date_list = [tarray[i]
                for i in range(number_entries)
                for k in range(number_lon)]



    # load coordinates
    # so height[i] is the geopotential height at a given time
    height = fileobj.variables['z'][:]
    g_inv = 1/9.81
    height = height*g_inv
    height = height / 9.81



    # get processed dataframe
    # Create storage object with filename `processed_data`
    name = 'processed_' + str(f[wantfile][-10:-5]) + '.h5'

    # Access data store
    data_store = pd.HDFStore(name)
    
    # Retrieve data using key
    geopot_df = data_store['preprocessed_geopot']
    data_store.close()



    #get fft coeffs
    def geopot_fft(geopotential):
        y = fft(geopotential)
        ck = y
        return(ck)

    fft_zonal_result = [geopot_fft(height[k]) for k in range(number_entries)]



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


# In[94]:




    def reptrend(i):
        sublist = fft_zonal_result[i]
        answer = [fwithZ(entry) for entry in sublist]
        return answer, i

    def temp_func(func, group): 
        return func(group)

    new_list = []

    def applyParallel(func): 
        answer, i = zip(*Parallel(n_jobs=mp.cpu_count())(delayed(temp_func)(func,group) for group in range(len(fft_zonal_result)))) 
        return new_list.append([answer, i])


    answer = applyParallel(reptrend)
    
    if len(new_list) == 1:
        tosplice = new_list[0]
    else:
        tosplice = new_list
        
    # get right amps
    unsortamps = np.array(tosplice[0])
    indices = np.array(tosplice[1])
    inds = indices.argsort()
    sortedamps = unsortamps[inds]
    
    amps = sortedamps[:,:,0]
    phases = sortedamps[:,:,1]
    
    # get indexing
    years = [item.year for item in tarray]
    s_months = [season_sort(item.month) for item in tarray]

    # pull together in a dataframe?
    d = {"index" : list(range(len(years))), "year": years, "season": s_months}

    index_df = pd.DataFrame(d)
    
    grouped_indices = index_df.groupby(["year","season"])   
    indices_only = grouped_indices["index"]
    indices_names = grouped_indices.apply(lambda x: x.name)
    
    #now splice up and then do a linear fit
    spliced = [[amps[k] 
                for k in list(indices_only.get_group(groupnm))]
               for groupnm in list(indices_names)]
    phases_grouped = [[phases[k] 
                       for k in list(indices_only.get_group(groupnm))]
               for groupnm in list(indices_names)]
    
    
    means = [np.mean(sublist, axis = 0) for sublist in spliced]
    test_answers = [np.polyfit(list(range(len(sublist))),sublist, deg = 1) 
                    for sublist in spliced]
    
    def arrayforsub(slope,intercept,N, meaned):
        #get the correct length 
        Narray = np.linspace(0, N, N)
        total_add = intercept - meaned
        tosub = np.add(slope*Narray, total_add)
        return tosub
    
    # get subtracted values from linear fit
    
    tosub = [[arrayforsub(slope = test_answers[k][0][j], 
                          intercept = test_answers[k][1][j], 
                          N = len(spliced[k]), meaned = means[k][j]) 
              for j in range(240)] 
             for k in range(len(spliced))]
    t_tosub = [np.transpose(sublist) for sublist in tosub]
    
    # get correct values
    adjusted_lists = [np.subtract(spliced[i], t_tosub[i]) 
                  for i in range(len(spliced))] 
    
    adjusted_totals = [[[adjusted_lists[group][day][lon]*np.exp(1j*phases_grouped[group][day][lon]) 
                         for lon in range(240)] 
                        for day in range(len(adjusted_lists[group]))] 
                       for group in range(len(adjusted_lists))]
    
    new_values = [[ifft(item) for item in sublist] for sublist in adjusted_totals]
    
    #make indexing dataframe
    lon_df_list = []
    year_df_list = []
    season_df_list = []
    testind = 0
    testdate_df = []


    for group_ind in range(len(spliced)): ## FIX!!
        for day_ind in range(len(spliced[group_ind])):
        
            maydate = tarray[testind]
            testind = testind + 1
        
            for lon_ind in range(len(amps[day_ind])):
            
        
                year = indices_names[group_ind][0]
                sea = indices_names[group_ind][1]
                lon = lon_list[lon_ind]
            
            
                lon_df_list.append(lon)
                year_df_list.append(year)
                season_df_list.append(sea)
                testdate_df.append(maydate)
    
    
    
    
    flat_values = flatten(flatten(new_values))
    
    # now bind it all together
    d = {"season": season_df_list, "year": year_df_list,
         "lon": lon_df_list, "adj_z": flat_values, "time" : testdate_df}

    untrend_df = pd.DataFrame(d)
    
    # also save frame untrended
    # Create storage object with filename `processed_data`
    data_store = pd.HDFStore('new_detrend_' + str(f[wantfile][-10:-5]) + '.h5')

    # Put DataFrame into the object setting the key as 'preprocessed_df'
    data_store['untrend_geopot'] = untrend_df
    data_store.close()



