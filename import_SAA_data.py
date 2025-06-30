#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:40:44 2025

@author: giuliosaibene
"""

# =============================================================================
# IMPORT SAA CONVERTED DATA AND EXTRACT VARIABLES FROM MATLAB FILES. 
# CALCULATE TOTAL DEFORMATAION VECTORS
# =============================================================================


import scipy.io
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

### DEFINE FUNCTIONS

# Extract x, y, z, time, and temp variables from MATLAB files

def extract_vars_mat(SAA_data):

    print(SAA_data.keys())
    
    cartesian_data = SAA_data['ArrayCartesian'][0, 0]['cart_data'][0, 0]
    
    # Extract the x, y, z fields
    x = cartesian_data['x'][0][0]  # x data
    y = cartesian_data['y'][0][0] # y data
    z = cartesian_data['z'][0][0] / 1000 - 40  # z data (converted to kilometers and offset)
    
    ms_time = cartesian_data['ms'][0][0]  # Time in milliseconds
    
    # Convert milliseconds to days, then add the base date (1980/1/1)
    base_date = datetime(1980, 1, 1)
    time = np.array([base_date + timedelta(milliseconds = int(ms)) for ms in ms_time])
    
    # Temperature
    
    temp_SAA = cartesian_data['tempc'][0][0]
    
    return [x, y, z, time, temp_SAA]


# convert to useful deformation values by subtracting initial value from all values
# returns individual relative components

def subtract_first_value(x, y, z, times, start_date):
    
    start_date = pd.to_datetime(start_date)
    
    tini = times[times.iloc[:, 0] == start_date].index[0]
            
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        
        # Subtract the first displacement for x values (sensor installation adjustment)
                
        x1 = x.subtract(x.loc[tini], axis=1)
        
        y1 = y.subtract(y.loc[tini], axis=1)
        
        z1 = z.subtract(z.loc[tini], axis=1)
        
    else:
        
        # Subtract the first displacement for x values (sensor installation adjustment)
        x1 = np.zeros_like(x)
        for i in range(len(x[:, 0])):
            x1[i, :] = x[i, :] - x[tini, :]
        
        # Subtract the first displacement for y values (sensor installation adjustment)
        y1 = np.zeros_like(y)
        for i in range(len(y[:, 0])):
            y1[i, :] = y[i, :] - y[tini, :]
            
        z1 = np.zeros_like(z)
        for i in range(len(z[:, 0])):
            z1[i, :] = z[i, :] - z[tini, :]
            
    # Merge time column with total deformation
    
    x1_only_df = pd.DataFrame(x1)
    y1_only_df = pd.DataFrame(y1)
    z1_only_df = pd.DataFrame(z1)
    
    x1_df = pd.concat([x1_only_df, times], axis = 1)
    y1_df = pd.concat([y1_only_df, times], axis = 1)
    z1_df = pd.concat([z1_only_df, times], axis = 1)
    
    x1_df.columns = x1_df.columns[:-1].tolist() + ['time']
    y1_df.columns = y1_df.columns[:-1].tolist() + ['time']
    z1_df.columns = z1_df.columns[:-1].tolist() + ['time']
            
    return [x1_df, y1_df, z1_df]


# Convert to deformation vectors and combine into total horizontal deformation
# Option to apply a 5-day smoothing window and resample to daily

def subtract_first_value_get_tot(x, y, times, start_date, one_year_switch,
                                 smoothSwitch, dailySwitch):
    
    start_date = pd.to_datetime(start_date)
    
    # End date exactly one year after
    end_date = start_date + pd.DateOffset(years=1)
    
    tini = times[times.iloc[:, 0] == start_date].index[0]
    
    print("Initial time:", tini)

        
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        
        # Subtract the first displacement for x values (sensor installation adjustment)
        x1 = x.subtract(x.loc[tini], axis=1)
        
        y1 = y.subtract(y.loc[tini], axis=1)
        
    else:
        
        # Subtract the first displacement for x values (sensor installation adjustment)
        x1 = np.zeros_like(x)
        for i in range(len(x[:, 0])):
            x1[i, :] = x[i, :] - x[tini, :]
        
        # Subtract the first displacement for y values (sensor installation adjustment)
        y1 = np.zeros_like(y)
        for i in range(len(y[:, 0])):
            y1[i, :] = y[i, :] - y[tini, :]

            
    # Total deformation
    #total_defo = np.sqrt(x1**2 + y1**2)
    total_defo = pd.DataFrame(np.sqrt(x1.to_numpy()**2 + y1.to_numpy()**2), 
                          index=x1.index, columns=x1.columns)


    print("total_defo", total_defo.head())
    
    if smoothSwitch:
        
        total_defo = total_defo.rolling(window = 15, min_periods = 1, center = True).mean() # five day window (8*15 / 24)
        
        total_defo_df = pd.DataFrame(total_defo)
            
    else:
        total_defo_df = pd.DataFrame(total_defo)
    
    # Merge time column with total deformation

    total_defo_df = pd.concat([total_defo_df, times], axis = 1)
    
    total_defo_df.columns = total_defo_df.columns[:-1].tolist() + ['time']
    
    # Resample to daily sums
    
    if dailySwitch:
    
        if 'time' in total_defo_df.columns:
        
            total_defo_df.set_index('time', inplace = True, drop = False)
        
        daily_defo_df = total_defo_df.resample('D').last()
            
        total_defo_df = daily_defo_df
    
    if one_year_switch:
        # Cut the total_defo_df to just one year long
        
        total_defo_df = total_defo_df[(total_defo_df['time'] >= start_date) & (total_defo_df['time'] < end_date)]
        
    return total_defo_df



# ------------------------------------- 

### IMPORT DATA

# Relative file paths to repository (make sure to set working directory to repository Murtel_2015_Data folder!!)

data_dir = "SAA_data" # subfolder 'SAA_data'

SAA_16_17_path = os.path.join(data_dir, "multi_saa_allcart_16_17.mat")
SAA_17_18_path = os.path.join(data_dir, "multi_saa_allcart_17_18.mat")
SAA_18_23_path = os.path.join(data_dir, "multi_saa_allcart_18_23.mat")

SAA_16_17 = scipy.io.loadmat(SAA_16_17_path)
SAA_17_18 = scipy.io.loadmat(SAA_17_18_path)
SAA_18_23 = scipy.io.loadmat(SAA_18_23_path)

# Extract variables

SAA_16_17_x, SAA_16_17_y, SAA_16_17_z, SAA_16_17_time, SAA_16_17_temp = extract_vars_mat(SAA_16_17)

SAA_17_18_x, SAA_17_18_y, SAA_17_18_z, SAA_17_18_time, SAA_17_18_temp = extract_vars_mat(SAA_17_18)

SAA_18_23_x, SAA_18_23_y, SAA_18_23_z, SAA_18_23_time, SAA_18_23_temp = extract_vars_mat(SAA_18_23)

# Combine data to 2016-18 period

SAA_16_18_x = pd.concat([pd.DataFrame(SAA_16_17_x), pd.DataFrame(SAA_17_18_x)], ignore_index = True)

SAA_16_18_y = pd.concat([pd.DataFrame(SAA_16_17_y), pd.DataFrame(SAA_17_18_y)], ignore_index = True)

SAA_16_18_z = pd.concat([pd.DataFrame(SAA_16_17_z), pd.DataFrame(SAA_17_18_z)], ignore_index = True)

SAA_16_18_time = pd.concat([pd.DataFrame(SAA_16_17_time), pd.DataFrame(SAA_17_18_time)], ignore_index = True)

SAA_16_18_temp = pd.concat([pd.DataFrame(SAA_16_17_temp), pd.DataFrame(SAA_17_18_temp)], ignore_index = True)

# Combine data to 2016-23 period

SAA_16_24_x = pd.concat([SAA_16_18_x, pd.DataFrame(SAA_18_23_x)], ignore_index = True)

SAA_16_24_y = pd.concat([SAA_16_18_y, pd.DataFrame(SAA_18_23_y)], ignore_index = True)

SAA_16_24_z = pd.concat([SAA_16_18_z, pd.DataFrame(SAA_18_23_z)], ignore_index = True)

SAA_16_24_time = pd.concat([SAA_16_18_time, pd.DataFrame(SAA_18_23_time)], ignore_index = True)

SAA_16_24_temp = pd.concat([SAA_16_18_temp, pd.DataFrame(SAA_18_23_temp)], ignore_index = True)

# Extract mean depths

mean_depths = abs(SAA_18_23_z.mean(axis = 0))

mean_depths_pd = pd.DataFrame(mean_depths)

# Get individual deformation vector components (used for 2D analysis)

SAA_16_24_x1, SAA_16_24_y1, SAA_16_24_z1 = subtract_first_value(SAA_16_24_x, SAA_16_24_y, 
                                                  SAA_16_24_z, SAA_16_24_time, 
                                                  start_date = "2016-01-06 12:05:00")

# Calculate total horizontal deformation

SAA_16_24_tot = subtract_first_value_get_tot(SAA_16_24_x, SAA_16_24_y, 
                                                  SAA_16_24_time, 
                                                  start_date = "2016-01-06 12:05:00", # start from beginning of data record
                                                  one_year_switch = False, # prevents data being cut to only last one year
                                                  smoothSwitch = True, # Applies 5 day moving window (n = 15)
                                                  dailySwitch = True) # Resamples to daily time resolution

### Export to CSV
SAA_16_24_path = os.path.join(data_dir, "SAA_16_24_tot.csv")
SAA_16_24_tot.to_csv(SAA_16_24_path, index=False)


# =============================================================================
# # OPTIONAL:
# # Clip data after 2016-04-01
# 
# if 'time' not in SAA_16_24_tot.columns:
# 
#     SAA_16_24_tot = SAA_16_24_tot[SAA_16_24_tot.index > pd.to_datetime("2016-04-01 00:00:00")]
# 
# else:
#     
#     SAA_16_24_tot = SAA_16_24_tot[SAA_16_24_tot['time'] > pd.to_datetime("2016-04-01 00:00:00")]
#     
# # Clip data before rockfall 2023-09-22
# 
# SAA_16_23_x1 = SAA_16_24_x1[SAA_16_24_x1['time'] < pd.to_datetime("2023-09-15 00:05:00")]
# SAA_16_23_y1 = SAA_16_24_y1[SAA_16_24_y1['time'] < pd.to_datetime("2023-09-15 00:05:00")]
# SAA_16_23_z1 = SAA_16_24_z1[SAA_16_24_z1['time'] < pd.to_datetime("2023-09-15 00:05:00")]
# 
# if 'time' not in SAA_16_24_tot.columns:
# 
#     SAA_16_23_tot = SAA_16_24_tot[SAA_16_24_tot.index < pd.to_datetime("2023-09-15 00:00:00")]
#     
# else:
#     
#     SAA_16_23_tot = SAA_16_24_tot[SAA_16_24_tot['time'] < pd.to_datetime("2023-09-15 00:00:00")]
# 
# 
# =============================================================================
