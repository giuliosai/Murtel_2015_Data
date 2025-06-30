#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:17:18 2025

@author: giuliosaibene
"""

import pandas as pd
import os

### DEFINE FUNCTIONS

# Get hydrological year starting in Sept 1
def get_hydrological_year(date):
    return date.year if date.month >= 9 else date.year - 1 # here defining 

# Function to interpolate only short gaps (< 3 days) for a single column
def interpolate_short_gaps(col, max_gap_days=2):
    # Identify NaN gaps
    is_nan = col.isna()
    gap_id = (is_nan != is_nan.shift()).cumsum() * is_nan  # Unique IDs for each NaN gap
    gap_lengths = col.groupby(gap_id).size()  # Get lengths of each gap
    
    # Identify short gaps (less than or equal to max_gap_days)
    short_gaps = gap_lengths[gap_lengths <= max_gap_days].index
    
    # Interpolate all gaps
    interpolated = col.interpolate(method='linear', limit_direction='forward', limit_area='inside')
    
    # Reset long gaps back to NaN
    col_filled = col.copy()
    col_filled = interpolated.where(gap_id.isin(short_gaps), col)  # Keep only short gaps filled
    
    return col_filled

# Function to parse depth from column name and for depths <4m to also fill gaps up to 14 days
def selective_filling_interpolation(col_name, col_data):
    # Extract depth from column name 
    if col_name == 'time' or col_name == 'hydro_year':
        return col_data
    
    depth = float(col_name)  # convert to float
    
    # Set max_gap_days based on depth
    if depth < 4:
        max_gap_days = 14
    else:
        max_gap_days = 2
    
    # Apply the interpolate_short_gaps function with the appropriate max_gap_days
    return interpolate_short_gaps(col_data, max_gap_days)

# ------------------------------------- 

### IMPORT DATA

# Relative file paths to repository (make sure to set working directory to repository Murtel_2015_Data folder!!)

data_dir = "CORtemp_data" # subfolder 'SAA_data'

temp_hole_87_path = os.path.join(data_dir, "COR_0287_temp.csv")
temp_hole_15_path = os.path.join(data_dir, "COR_0315_temp.csv")

temp_hole_87 = pd.read_csv(temp_hole_87_path)
temp_hole_15 = pd.read_csv(temp_hole_15_path)

temp_hole_87['time'] = pd.to_datetime(temp_hole_87['time'])
temp_hole_15['time'] = pd.to_datetime(temp_hole_15['time'])

temp_hole_87['hydro_year'] = temp_hole_87['time'].apply(get_hydrological_year)
temp_hole_15['hydro_year'] = temp_hole_15['time'].apply(get_hydrological_year)

# Data gap filling by interpolation

# Interpolate to fill gaps shorter than 3 days for depths > 4m and gaps shorter than 14 days for depths < 4m
temp_hole_87_interpolated = pd.DataFrame({
    col_name: selective_filling_interpolation(col_name, temp_hole_87[col_name])
    for col_name in temp_hole_87.columns
})

temp_hole_15_interpolated = pd.DataFrame({
    col_name: selective_filling_interpolation(col_name, temp_hole_15[col_name])
    for col_name in temp_hole_15.columns
})

# Start 1987 borehole data from 1994 where continuous data starts:
    
temp_hole_87 = temp_hole_87_interpolated[temp_hole_87_interpolated['time'].dt.year >= 1994]

# End 2015 borehole data to end of 2023

temp_hole_15 = temp_hole_15_interpolated[temp_hole_15_interpolated['time'].dt.year <= 2023]

### Export to CSV

temp_hole_87_export_path = os.path.join(data_dir, "temp_hole_87.csv")
temp_hole_87.to_csv(temp_hole_87_export_path, index=False)

temp_hole_15_export_path = os.path.join(data_dir, "temp_hole_15.csv")
temp_hole_15.to_csv(temp_hole_15_export_path, index=False)

