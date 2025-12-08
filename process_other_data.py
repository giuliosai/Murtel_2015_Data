#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:54:17 2025

@author: giuliosaibene
"""

# =============================================================================
# IMPORT REST OF THE DATA: GNSS, geodetic, and meteo
# =============================================================================


import scipy.io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

## DEFINE FUNCTIONS USED IN THE PROCESS


# Get hydrological year starting in Sept 1
def get_hydrological_year(date):
    return date.year if date.month >= 9 else date.year - 1

# Convert dates to julian days to be able to subset each year according to the snow_onset_time

def get_julian_day(date):
    
    if pd.isna(date):
        
        return None
    
    date = pd.to_datetime(date)
    
    year_start = datetime(date.year, 1, 1)
    
    return (date - year_start).days + 1


# Define the julian day = 1 as September 1 based on the used definition of an hydrological year

def custom_julian_day(date):
    
    if pd.isna(date):
        
        return None
    
    year_start = datetime(date.year, 9, 1)
    
    if date < year_start:
        year_start = datetime(date.year - 1, 9, 1)
        
    return (date - year_start).days 

## Prepare velocity data to be plotted in annual step lines

def expand_velocity_data(annual_defo):
    expanded_dates = []
    expanded_velocities = []

    for i in range(len(annual_defo)):
        # Ensure we don't loop back from first to last row
        if i == 0:
            continue
        
        if 'survey_date' in annual_defo.columns:

            start = annual_defo['survey_date'].iloc[i - 1]
            end = annual_defo['survey_date'].iloc[i]
            
        elif 'date' in annual_defo.columns:
            
            start = annual_defo['date'].iloc[i - 1]
            end = annual_defo['date'].iloc[i]
            
        else:
            
            print("Date column not found in input data")

        if 'vel2d' in annual_defo.columns:
            
            # Add the start point (with vel2d value from the current row)
            expanded_dates.append(start)
            expanded_velocities.append(annual_defo['vel2d'].iloc[i])
    
            # Add the end point (with the same vel2d value)
            expanded_dates.append(end)
            expanded_velocities.append(annual_defo['vel2d'].iloc[i])
    
        
        elif 'horiz_displacement_filt' in annual_defo.columns:
            
            # Add the start point (with vel2d value from the current row)
            expanded_dates.append(start)
            expanded_velocities.append(annual_defo['horiz_displacement_filt'].iloc[i])
    
            # Add the end point (with the same horiz_displacement_filt value)
            expanded_dates.append(end)
            expanded_velocities.append(annual_defo['horiz_displacement_filt'].iloc[i])
            
        else:
            
            print("No deformation column found in data")
        
        # Convert expanded dates and velocities into a DataFrame for plotting
        plot_df = pd.DataFrame({'date': expanded_dates, 'defo': expanded_velocities})
        plot_df['date'] = pd.to_datetime(plot_df['date'])

    return plot_df

## Data gap interpolation

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

# Aggregate to annual means:

def aggregate_to_annual_from_daily(df, sumSwitch, valid_days_switch):
    
    """
    
    Computes calendar year mean, min, max, and when min and max happen
    
    sumSwitch = True --> Computes hydro year sums (useful for snow depth)
    
    Else it just groups by calendar year
    
    """
    
    if 'date' in df.columns:
    
        df['year'] = df['date'].dt.to_period('Y')
        timestamp_column = 'date'
        
    elif 'time' in df.columns:
        df['year'] = df['time'].dt.to_period('Y')
        timestamp_column = 'time'
        
    elif isinstance(df.index, pd.DatetimeIndex):
        
        df['year'] = df.index.year.values
        df['time'] = df.index.values
        timestamp_column = 'time'

    else:
        print("Neither 'date' or 'time' are columns in dataframe")
    
    if 'season' in df.columns:
        df_clean = df.drop('season', axis=1)
        
    else:
        df_clean = df
    
    if valid_days_switch:
    
        # Group by year and count the number of months with available data
        day_counts_per_year = df_clean.groupby('year').size()
        
        # Consider years with at least 12 months with data
        valid_years = day_counts_per_year[day_counts_per_year >= 360].index
        
        # Keep only the years with enough data
        df_valid_years = df_clean[df_clean['year'].isin(valid_years)]
        
    else:
        
        day_counts_per_year = 365
        
        df_valid_years = df_clean
    
    if sumSwitch: # To get hydro year sums of snow depth
        
        # Check if there is 'hydro_year' column
        
        if 'hydro_year' in df.columns:
        
            # Compute annual sums by only considering numeric columns
            annual_means = df_valid_years.groupby('hydro_year').agg(lambda x: x.sum() if x.dtype in ['int64', 'float64'] else x.iloc[0])
            
        else:
            
            df_valid_years['hydro_year'] = df_valid_years['year'].apply(get_hydrological_year)
            
            # Compute annual sums by only considering numeric columns
            annual_means = df_valid_years.groupby('hydro_year').agg(lambda x: x.sum() if x.dtype in ['int64', 'float64'] else x.iloc[0])
    
    else:
        # Compute annual means by only considering numeric columns
        annual_means = df_valid_years.groupby('year').agg(lambda x: x.mean() if x.dtype in ['int64', 'float64'] else x.iloc[0])
    
    annual_sd = df_valid_years.groupby('year').agg(lambda x: x.std() if x.dtype in ['int64', 'float64'] else x.iloc[0])

    # Compute Standard Error of the Mean (SEM)
    
    # Loop through each numeric column to compute SEM and add a new column dynamically
    for col in annual_means.select_dtypes(include=['int64', 'float64']).columns:
        
        annual_means[f'sd_{col}'] = annual_sd[col]
        
        annual_means[f'sem_{col}'] = annual_sd[col] / np.sqrt(day_counts_per_year)
    
    annual_max = df_valid_years.groupby('year').agg(
        lambda x: x.max() if x.dtype in ['int64', 'float64'] else x.iloc[0])
    
    annual_min = df_valid_years.groupby('year').agg(
        lambda x: x.min() if x.dtype in ['int64', 'float64'] else x.iloc[0])
    
    # Extract the timestamps for max and min
    # Selects only numeric columns, finds at what index the max and min occur and then selects the value of the timestamp_column
    # where (.loc) the cell value corresponds with the index where the max value is found
    
    max_indices = df_valid_years.groupby('year').idxmax()
    
    max_timestamp_df = pd.DataFrame(index=max_indices.index, columns=max_indices.columns)

    for col in max_indices.columns:
        for year in max_indices.index:  # Loop through each row (year)
            idx = max_indices.loc[year, col]
            if pd.isna(idx):
                # If the index is NaN, keep NaN in the max_timestamp_df
                max_timestamp_df.loc[year, col] = pd.NA
            else:
                # Otherwise, get the timestamp corresponding to the valid index
                max_timestamp_df.loc[year, col] = df_valid_years[timestamp_column].loc[idx]
                
    
    min_indices = df_valid_years.groupby('year').idxmin()
    
    min_timestamp_df = pd.DataFrame(index=min_indices.index, columns=min_indices.columns)

    for col in min_indices.columns:
        for year in min_indices.index:  # Loop through each row (year)
            idx = min_indices.loc[year, col]
            if pd.isna(idx):
                # If the index is NaN, keep NaN in the min_timestamp_df
                min_timestamp_df.loc[year, col] = pd.NA
            else:
                # Otherwise, get the timestamp corresponding to the valid index
                min_timestamp_df.loc[year, col] = df_valid_years[timestamp_column].loc[idx]

    
    return [annual_means, annual_min, annual_max, min_timestamp_df, max_timestamp_df]


def plot_contours_temp_ALT(temp_df, depths_str, plot_switch):
    
    """
    Use the contour plot matplotlib function to interpolate to find the
    0˚C isotherm depths to extract the ALT from the borehole temperatures.
    Returns the yearly maximum ALT, the date of maximum ALT, the julian day
    of maximum ALT, a mean ALT five time steps before and after the date of
    max ALT, a SD of the ALT across the same time steps as the mean.
    """
    
    times = pd.to_datetime(temp_df['time'])
    depths = temp_df.columns[1:-1] # don't take first and last columns - this is a pandas series
    
    # Convert to list of floats
    depth_values = [float(depth.strip("'")) for depth in depths_str.split(', ')] 
    
    # Convert to list of strings to subset dataframe
    depth_values_str = [str(depth.strip("'")) for depth in depths_str.split(', ')] 
    
    # Convert to pandas series
    depth_values_ser = pd.Series(depth_values_str)
    
    # Transpose such that rows are depths and columns are time points
    temperature_depth_array = temp_df[depth_values_ser].values.T
    
    # Create the contour plot
    fig, ax = plt.subplots(figsize=(10, 6))
    time_values = np.arange(len(times))
    
    # Create a meshgrid for contour plot
    time_mesh, depth_mesh = np.meshgrid(time_values, depth_values)
    
    #print("times:",times)
    
    cmap_colors = [(0.0, '#002d70'), (0.25, "#649bed"), (0.5, 'white'), (0.75, "#ff6652"), (1.0, '#bf1600')]
    cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors, N=256)
    cmap_custom.set_under(color='gray')  # Assign black to values below the colormap range
    
    temp_min = np.nanmin(temperature_depth_array)
    temp_max = np.nanmax(temperature_depth_array)

    contour = ax.contourf(time_mesh, depth_mesh, 
                          temperature_depth_array, cmap = 'RdBu_r',
                          vmin = temp_min - 5,
                          vmax = temp_max)
    
    zero_contour = ax.contour(time_mesh, depth_mesh, temperature_depth_array, levels=[0], colors='black')
    
    fig.colorbar(contour)
    
    ax.set_xticks(time_values)
    ax.set_xticklabels(times.dt.strftime('%Y-%m'), rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.set_xlabel('Time')
    ax.set_ylabel('Depth (m)')
    #ax.set_ylim(0,5) # look at only top x m
    ax.invert_yaxis()
    
    # Plot contour lines
    ax.clabel(zero_contour, inline=True, fontsize=8)  # Add labels to the contour lines

    # Extract the paths from the contour lines to get the 0˚C isotherm
    paths = zero_contour.collections[0].get_paths()
    
    # Find x and y points of this isotherm
    
    zero_temperature_points = []
    for path in paths:
        vertices = path.vertices
        x = vertices[:, 0]
        y = vertices[:, 1]
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        x_interp = np.linspace(min(x), max(x), num=1000)  # Increase num for higher resolution
        y_interp = f(x_interp)
        zero_temperature_points.extend(list(zip(x_interp, y_interp)))

    # Array of x and y values of 0˚ contour (y values are the depth values)
    zero_temperature_points = np.array(zero_temperature_points)
    
    zero_isotherm_x = pd.Series(zero_temperature_points[:,0]) # indices to match up with times
    zero_isotherm_x = zero_isotherm_x.astype(int)
    
    zero_isotherm_depths = pd.Series(zero_temperature_points[:,1])
    
    #print(zero_isotherm_depths)
    
    # Visually check depths of 0˚C isotherms
    
    if plot_switch:
    
        plt.figure()
        plt.scatter(zero_isotherm_depths.index, zero_isotherm_depths, s = 5) # These are the depth values
    
        plt.show()
        
    # Extract daily time series of zero isotherms
    
    times_ordered = times.reset_index(inplace = False, drop = True) # 0 - 2679

    isotherm_df = pd.DataFrame(index=times_ordered.index)  # Start with the same index as times
    isotherm_df['depth'] = np.nan
    isotherm_df.iloc[zero_isotherm_x, 0] = zero_isotherm_depths
    isotherm_df['time'] = times

    # Find yearly maximum depth of zero-th isotherm to also include years that 
    # are not captured by the plateau method above
    
    years = temp_df['time'].dt.year.unique()
    
    year_size = len(zero_isotherm_depths) / len(years)

    # Create a group identifier for each 200-period chunk
    group_id = (zero_isotherm_depths.index // year_size)

    # Calculate the maximum value for each period chunk
    ALT_yearly_max = zero_isotherm_depths.groupby(group_id).max()
    max_indices = zero_isotherm_depths.groupby(group_id).idxmax()
    
    # Find time when max occurs
    
    # Index the dates in times by the "x" values of the paths from the 0˚C isotherm which correspond with index values
    
    time_idx_maxs = zero_temperature_points[max_indices,0]
    
    times_maxs = [times.iloc[int(idx)] for idx in time_idx_maxs]
        
    # convert times to day of the year
    
    julian_day_maxs = [get_julian_day(date) for date in times_maxs]
        
    # Initialize lists to store the means and standard deviations of values around max value
    means = []
    std_devs = []
    
    # Loop over each index where the max occurs
    for idx in max_indices:
        # Define the range (5 indices before and 5 indices after, including the max index)
        start_idx = max(0, idx - 3)
        end_idx = min(len(zero_isotherm_depths), idx + 3)
        
        # Extract the values around the max index
        surrounding_values = zero_isotherm_depths.iloc[start_idx:end_idx + 1]
        
        # Calculate the mean and standard deviation
        means.append(surrounding_values.mean())
        std_devs.append(surrounding_values.std())
    
    # Calculate the percentiles for each group
    percentiles = zero_isotherm_depths.groupby(group_id).quantile([0.85, 0.90, 0.95])

    # Reshape the percentiles DataFrame so that the percentile values are columns
    percentiles = percentiles.unstack(level=-1)
    percentiles.columns = [f'ALT_{int(p*100)}th' for p in percentiles.columns]
    
    ALT_yearly_max_df = pd.DataFrame({
        'year': years,
        'ALT_max': ALT_yearly_max,
        'ALT_max_date': times_maxs,
        'ALT_max_julian_day': julian_day_maxs,
        'mean_around_max': means,
        'std_around_max': std_devs
    }).reset_index(drop=True)
    
    # Add the percentile columns to the DataFrame
    ALT_yearly_max_df = pd.concat([ALT_yearly_max_df, percentiles.reset_index(drop=True)], axis=1)
    
    # Add column for "uncertainty" by subtracting max value by 75th percentile
    ALT_yearly_max_df['ALT_max_to_95th'] = ALT_yearly_max_df['ALT_max'] - ALT_yearly_max_df['ALT_95th']
    
    return ALT_yearly_max_df, isotherm_df

def find_freezing_start_temp(df, smoothSwitch):
    
    """
    Based on the ground temperature measured by the uppermost thermistor
    in the borehole chain the first date when the temperature becomes
    consistently below 0˚ is found.
    A smoothSwitch is applied if a 10-day moving window average is desired.
    """
    
    freezing_start_days = []
    
    if smoothSwitch:
        
        df_copy = df.copy()
        
        for year in df_copy['hydro_year'].unique():
            
            if '0.5' in df_copy.columns:
                
                df_copy['0.5_smooth'] = df_copy['0.5'].rolling(window = 10).mean()
            
                freezing_year = df_copy[(df_copy['hydro_year'] == year) & (df_copy['0.5'] < 0)]
            
            else:
                
                df_copy['0.55_smooth'] = df_copy['0.55'].rolling(window = 10).mean()
                
                freezing_year = df_copy[(df_copy['hydro_year'] == year) & (df_copy['0.55_smooth'] < 0)]
            
            if not freezing_year.empty:
                first_freezing_day = freezing_year['time'].iloc[0]
                                
                freezing_start_days.append({'hydro_year': year, 'first_freezing_day': first_freezing_day})
                
            else:
                freezing_start_days.append({'hydro_year': year, 'first_freezing_day': np.nan})
                
        result_df = pd.DataFrame(freezing_start_days)
        
    else:
    
        for year in df['hydro_year'].unique():
            
            if '0.5' in df.columns:
            
                freezing_year = df[(df['hydro_year'] == year) & (df['0.5'] < 0)]
            
            else:
                
                freezing_year = df[(df['hydro_year'] == year) & (df['0.55'] < 0)]
            
            if not freezing_year.empty:
                first_freezing_day = freezing_year['time'].iloc[0]
                                    
                freezing_start_days.append({'hydro_year': year, 'first_freezing_day': first_freezing_day})
    
                
            else:
                freezing_start_days.append({'hydro_year': year, 'first_freezing_day': np.nan})
                
        result_df = pd.DataFrame(freezing_start_days)
    
    return result_df


# Getting annual means just from values from Nov - Jan

def early_winter_annual_mean(df):
    
    if 'season' in df.columns:
    
        df = df.drop('season', axis=1)
    
    # Define the early winter
    early_winter = [11,12,1]
    
    df['month'] = df['date'].dt.month
    df['hydro_year'] = df['date'].apply(get_hydrological_year)
    
    df_early_winter = df[df['month'].isin(early_winter)]
    
    days_per_early_winter = df_early_winter.groupby('hydro_year').size()
    
    # Group by season and compute seasonal means
    
    early_winter_means_from_daily = df_early_winter.groupby('hydro_year', as_index=False).agg(lambda x: x.mean() if x.dtype in ['int64', 'float64'] else x.iloc[0])
    
    early_winter_sd = df_early_winter.groupby('hydro_year', as_index=False).agg(lambda x: x.std() if x.dtype in ['int64', 'float64'] else x.iloc[0])
    
    for col in df_early_winter.select_dtypes(include=['int64', 'float64']).columns:
        
        early_winter_means_from_daily[f'sd_{col}'] = early_winter_sd[col]
        
        early_winter_means_from_daily[f'sem_{col}'] = early_winter_sd[col] / np.sqrt(days_per_early_winter.iloc[1])

    return df_early_winter, early_winter_means_from_daily

def find_last_snow_date(data):
    # Initialize variables to track the start of the zero snow period
    start_date = None
    current_period_days = 0
    last_snow_date = None
                
                
    for index, row in data.iterrows():
        
        if abs(row['snowh']) <= 0.06: # check is snow is within ±0.06m which seems to be the measurement precision
            if start_date is None:
                # Start of a new zero snow period
                start_date = row['date']
                current_period_days = 1
            
    
            else:
                # Continuation of the zero snow period
                current_period_days += 1
        else:
            # Snow height is non-zero, reset the period
            start_date = None
            current_period_days = 0

    
        # Check if the current period meets the criteria
        if current_period_days > 14:
            last_snow_date = start_date - pd.Timedelta(days=1)
            break

    return last_snow_date


def get_snow_end_time(df):
    
    # Convert to using hydrological years
    
    # Add a hydrological year column
    df['hydro_year'] = df['date'].apply(get_hydrological_year)
    
    # Identify the range of hydrological years
    min_year = df['hydro_year'].min()
    max_year = df['hydro_year'].max()
    
    # Get all years to fill in missing years
    all_years = pd.DataFrame({'HydroYear': range(min_year, max_year + 1)})

    
    # Group by hydrological year and find the last date for each hydrological year
    last_snow_dates = []
    
    for hydro_year, group in df.groupby('hydro_year'):
        last_snow_date = find_last_snow_date(group)
        if last_snow_date is not None:
            last_snow_dates.append({'HydroYear': hydro_year, 'LastSnowDate': last_snow_date})
    
    # Convert the result to a DataFrame
    result_df = pd.DataFrame(last_snow_dates)
    
    # Merge with all_years to ensure all years are represented
    result_df = pd.merge(all_years, result_df, on='HydroYear', how='left')

    # Set LastSnowDate to default value for missing years 
    # (only 2019 is missing and so I have set this date to the start of the spring ZC for 2019)
    result_df['LastSnowDate'] = result_df['LastSnowDate'].fillna(
        result_df['HydroYear'].apply(lambda x: f"{x + 1}-06-15")
    )

    return result_df


def get_phase_out_of_date(df, freezing_time, snow_end_time):
    
    """
    Give a series of dates it assigns each of them to either a warm phase or cold phase
    
    The date thresholds used are the end of snow cover and freezing time at 0.5 m in borehole
    - For these the julian date hydro (so counting dates from start of hydrological year) is used
    
    It loops over every hydrological year available in the snow_end_time data
    
    For each hydro year it creates two masks:
    - A cold mask for Julian hydro days larger than the freezing Julian hydro day (has a mean of 50 DOY) 
    AND lower than the snow end Julian hydro day (has a mean of 286 DOY)
    - A warm mask for Julian hydro days lower than the freezing Julian hydro day (has a mean of 50 DOY)
    OR larger than the snow end Julian hydro day (has a mean of 286 DOY)
    
    
    """
    
    pd.options.mode.chained_assignment = None  # default='warn', get rid of not very useful warnings
    
    # Ensure date columns are in datetime format
    freezing_time['first_freezing_day'] = pd.to_datetime(freezing_time['first_freezing_day'])
    snow_end_time['LastSnowDate'] = pd.to_datetime(snow_end_time['LastSnowDate'])
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    elif 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
        
    else:
        df['date'] = pd.to_datetime(df.index)
    
    # Get Julian day of snow onset date
    
    freezing_time['julian_day_hydro'] = freezing_time['first_freezing_day'].apply(custom_julian_day)
    
    # Get julian day of end date of spring ZC
    
    snow_end_time['julian_day_hydro'] = snow_end_time['LastSnowDate'].apply(custom_julian_day)
    
    if 'julian_day_hydro' not in df.columns:
        # Get julian day (custom meaning starting from Sept 1)
        df['julian_day_hydro'] = df['date'].apply(custom_julian_day)
        
    # Making sure hydro_year is a column
    
    if 'HydroYear' in snow_end_time.columns:
        snow_end_time['hydro_year'] = snow_end_time['HydroYear']
    
    df['hydro_year'] = df['date'].apply(get_hydrological_year)
    
    print(df['date'].head())
    
    # Initialize 'phase' column
    df.loc[:, 'phase'] = 'Unknown'
    
    # Cold phase starts on start_date_cold (snow onset time) and ends on end_date_cold (end of spring ZC)
    
    # Warm phase starts at end_date_cold (end of spring ZC) and ends at start_date_cold (freezing_time)
    
    # Create a 'phase' column to write out based on the julian day if it is in the cold or warm phase
    
    # Loop over hydro year to define phase
    
    df_list = []
    
    # Define list of available years in the main df input
    hydro_years_df = df['hydro_year'].unique()
    
    print("Years in df:", hydro_years_df)
    print("Years in snow_end_time:", snow_end_time['hydro_year'].unique())

    
    for year in snow_end_time['hydro_year']: # these are only years from the borehole time period used to get freezing times
        
        # Handling cases when the year from the freezing_time data frame is not found in the main input dataframe
        if year not in hydro_years_df:
            print(f'Skipping year {year} as it is not present in the input dataframe')
            
            continue 
        
        # Select rows in the dataframe that match the current hydrological year
        df_year = df[df['hydro_year'] == year]
        
        # Filter snow onset data for the current hydrological year
        freezing_day_data = freezing_time[freezing_time['hydro_year'] == year]
                
        # Also in case that there is an unrealistic freezing day due to a data gap in the borehole temp series
        if freezing_day_data.empty or (freezing_day_data['julian_day_hydro'] > 100).any():
            freezing_date_hydro = 48 # Set to mean julian day hydro for freezing date from 1987 data series
            
        else:
            # Get the start of the cold phase (snow onset)
            freezing_date_hydro = freezing_day_data['julian_day_hydro'].values[0]
        
        # Filter spring ZC data for the current hydrological year
        snow_end_time_data = snow_end_time[snow_end_time['hydro_year'] == year]
        
        # This is a problem since for 2001-02 there is no spring ZC data
        if snow_end_time_data.empty:
            snow_end_date_hydro = 287 # In case it's missing set to mean of all snow end date values (June 15)
            #continue  # Skip this year if no spring ZC data is found

        else:
            
            # Get the end of the cold phase (end of spring ZC)
            snow_end_date_hydro = snow_end_time_data['julian_day_hydro'].values[0]
        
        # Assign 'Cold' phase for dates between snow onset and the end of spring ZC
        cold_mask = (df_year['julian_day_hydro'].copy() >= freezing_date_hydro) & (df_year['julian_day_hydro'].copy() <= snow_end_date_hydro)
        df_year.loc[cold_mask, 'phase'] = 'Cold'
        df_year.loc[cold_mask, 'period_start_date'] = freezing_date_hydro
        df_year.loc[cold_mask, 'period_end_date'] = snow_end_date_hydro
        
        # Warm phase: Outside the cold period in the same hydrological year
        warm_mask = (df_year['julian_day_hydro'].copy() < freezing_date_hydro) | (df_year['julian_day_hydro'].copy() > snow_end_date_hydro)
        df_year.loc[warm_mask, 'phase'] = 'Warm'
        df_year.loc[warm_mask, 'period_start_date'] = snow_end_date_hydro
        df_year.loc[warm_mask, 'period_end_date'] = freezing_date_hydro
        
        df_list.append(df_year.copy())
        
    df = pd.concat(df_list, ignore_index = True)
    
    df['phase_id'] = (df['phase'] != df['phase'].shift()).cumsum()
    
    # Calculate phase duration in days for each phase_id group
    phase_duration = df.groupby('phase_id')['date'].agg(lambda x: (x.max() - x.min()).days)
    
    df['phase_duration'] = df['phase_id'].map(phase_duration)
    
    df_warm = df[df['phase'] == 'Warm' ]
    
    df_cold = df[df['phase'] == 'Cold']

    return df, df_warm, df_cold

def find_constant_periods(temperature_series, step_tolerance, period_tolerance):
    
    """
    Given a borehole temperature series and two tolerances periods
    when the temperature is not changing and is around 0˚C are extracted.
    - Step tolerance: decides how much change in temperature between individual
    time steps is needed to end the constant period.
    - Period tolerance: decides how much change is needed across the entire
    given period is needed to end it.
    """
    
    constant_periods = []
    const_period_start_index = 0
    in_constant_period = False
    
    # Reset index of the input
    temperature_series = temperature_series.reset_index(drop = True) # Drop index column because I only want the temp column

    for i in range(1, len(temperature_series)):
        
        # Check if the value from point i to the start of the constant period is within the tolerance,
        # then start a constant period and set the constant_period_start_index to that day
        
        if abs(temperature_series[i] - temperature_series[const_period_start_index]) <= period_tolerance:
            if not in_constant_period:
                in_constant_period = True
                const_period_start_index = i - 1 # converting i values to index 
                # ( i = 1 when index = 0 )
            # So if you are in constant period still then the const_period_start_index 
            # does not change and remains the original start date
        
        
        # If the tolerance check relative to the start of the period does not pass,
        # then check whether the temperature has changed enough relative to the previous day to really end the period
        
        else: 
            
            # End period only if the step tolerance is exceeded or in cases where even if the step tolerance 
            # hasn't been exceeded the temperature has gradually shifted far enough away from the starting temp
            if in_constant_period and (
                abs(temperature_series[i] - temperature_series[i-1]) >= step_tolerance or
                abs(temperature_series[i] - temperature_series[const_period_start_index]) >= 1.5 * period_tolerance
                ):
                
                # Only keep pairs of dates when they are far apart enough from each other (14 days)
                if (i-1) - const_period_start_index > 14:
                
                    # this is the end of the constant period and so append the tuple of indices to the list
                    constant_periods.append((const_period_start_index, i - 1))
                
                # Say that we are not in a constant period anymore and 
                # go back to check whether the first tolerance check is passed
                in_constant_period = False
                
                # Reset the const_period_start_index here,
                # so that the next tolerance test does not still use the index from the start of the previous period
                const_period_start_index = i - 1
            
            # If it doesn't pass the first tolerance check and is not in a constant period
            # i.e. it hasn't found the first constant period yet, keep shifting the const_period_start_index so it doesn't stay at 0
            if not in_constant_period:
                const_period_start_index = i - 1

    # Check if the last period is constant till the end of the series
    if in_constant_period:
        constant_periods.append((const_period_start_index, len(temperature_series) - 1))

    # The returned list of tuple pairs is based on the reset index so starts from 0
    return constant_periods

# Function to ge the whole data series long version to plot each point of each ZC

def get_constant_period_data(df, constant_periods):
    
    """
    Given a dataframe (df) with the borehole temperature data and the 
    output from the find_constant_periods() function it outputs a dataframe
    with all the dates and temperatures of the constant periods.
    """
    
    ZCs_date_list = []
    ZCs_temp_list = []
    
    # Make sure the index of the input data also starts from 0
    df = df.reset_index(drop = True)

    for start, end in constant_periods:
        # the reset_index() is to add an index column to be able to merge the two later, 
        # it's not actually changing the index as it already starts at 0
        ZC_date = df.iloc[start:end+1, 0].reset_index() 
        ZC_temp = df.iloc[start:end+1, 1].reset_index()
        ZCs_date_list.append(ZC_date)
        ZCs_temp_list.append(ZC_temp)
        
        ZCs_dates = pd.concat(ZCs_date_list)
        ZCs_temps = pd.concat(ZCs_temp_list)
        
        ZCs_df = pd.merge(ZCs_dates, ZCs_temps)
        
        # Select only periods when temperature is around zero:
        ZCs_df_filtered = ZCs_df[(ZCs_df.iloc[:, 2] >= -0.1) & (ZCs_df.iloc[:, 2] <= 0.1)]

    return ZCs_df_filtered

# Function to get the duration, start date and the end date of the ZC

def get_start_end_duration(df, constant_periods):
    
    """
    Given the borehole temperature dataframe (df) and the constant_periods,
    the start and end dates of each period are the output. The duration is 
    also calculated here.
    """
    
    ZCs_duration = [b - a for a, b in constant_periods]
    
    ZCs_start_dates = []
    ZCs_end_dates = []
    
    for start, end in constant_periods:
        
        ZCs_start_date = df['time'].iloc[start]
        ZCs_end_date = df['time'].iloc[end]
        
        ZCs_start_dates.append(ZCs_start_date)
        ZCs_end_dates.append(ZCs_end_date)
    
    # Create a DataFrame from the lists
    ZCs_times_df = pd.DataFrame({
        'ZC_duration': ZCs_duration,
        'start_date': ZCs_start_dates,
        'end_date': ZCs_end_dates,
        'year': pd.to_datetime(ZCs_start_dates).year
    })
    
    ZCs_times_df['start_date'] = pd.to_datetime(ZCs_times_df['start_date'])
    ZCs_times_df['end_date'] = pd.to_datetime(ZCs_times_df['end_date'])
    
    # Filter to keep only periods that are actually during a reasonable ZC time
    spring_months = [5, 6]
    autumn_months = [9,10,11]
    spring_ZCs_times_df = ZCs_times_df[ZCs_times_df['start_date'].dt.month.isin(spring_months)]
    autumn_ZCs_times_df = ZCs_times_df[ZCs_times_df['start_date'].dt.month.isin(autumn_months)]
    
    # Drop duplicates based on 'year', keeping only the first row
    autumn_ZCs_times_df = autumn_ZCs_times_df.drop_duplicates(subset=['year'], keep='first')
    spring_ZCs_times_df = spring_ZCs_times_df.drop_duplicates(subset=['year'], keep='first')
        
    return [spring_ZCs_times_df, autumn_ZCs_times_df]



def get_layer_specific_mean(depth_min, depth_max, temp_hole):

    # Only select numeric columns
    layer_columns = [
                col for col in temp_hole.columns 
                if col.replace('.', '', 1).isdigit()  # Checks if the column name can be treated as a number
                and depth_min <= float(col) <= depth_max
                    ]
    
    temp_hole_layer_mean = temp_hole[layer_columns].mean(axis = 1)
    temp_hole_layer_sd = temp_hole[layer_columns].std(axis = 1)
    
    return [temp_hole_layer_mean, temp_hole_layer_sd]

# Aggregate to daily means

def aggregate_to_daily(df):

    if 'date' in df.columns:
        # Use 'date' column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_only'] = df['date'].dt.date
        #date_column = 'date_only'
    elif 'time' in df.columns:
        # Create 'date_only' from 'time' if 'time' column exists
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['date_only'] = pd.to_datetime(df['time']).dt.date
        #date_column = 'date_only'
    else:
        # Return error message if neither column exists
        return "No relevant column found"
    
    # Group by date and count the number of hours with available data to only then keep days with at least 19 h
    hour_counts_per_day = df.groupby('date_only').size()
    
    valid_dates = hour_counts_per_day[hour_counts_per_day >= 15].index
    
    # Keep only the days with enough data
    df_valid_days = df[df['date_only'].isin(valid_dates)]
    
    numeric_cols = df_valid_days.select_dtypes(include=['object']).columns.difference(['time', 'date_only'])

    df_valid_days[numeric_cols] = df_valid_days[numeric_cols].apply(pd.to_numeric, errors='coerce')

    daily_means = df_valid_days.groupby('date_only').mean()
    
    return daily_means


# -------------- Import data --------------------- #

def load_and_prepare_gnss_data(data_dir = "GNSS_data"):
    
    gnss_diff_daily_path = os.path.join(data_dir, "COR1_gps_differential_daily_diff.csv")
    
    gnss_diff_monthly_path = os.path.join(data_dir, "COR1_gps_differential_monthly_diff.csv")
    
    gnss_raw_incli_path = os.path.join(data_dir, "COR1_gps_inclinometer.csv")
        
    gnss_diff_daily = pd.read_csv(gnss_diff_daily_path)
    
    gnss_diff_monthly = pd.read_csv(gnss_diff_monthly_path)  
    
    gnss_raw_incli = pd.read_csv(gnss_raw_incli_path)
    
    gnss_diff_daily['time'] = pd.to_datetime(gnss_diff_daily['time'], errors='coerce')
    
    gnss_diff_monthly['time'] = pd.to_datetime(gnss_diff_monthly['time'], errors='coerce')
    
    gnss_diff_daily_23 = gnss_diff_daily[gnss_diff_daily['time'] <= pd.to_datetime("2023-08-27")]
    
    gnss_diff_monthly_23 = gnss_diff_monthly[gnss_diff_monthly['time'] <= pd.to_datetime("2023-08-27")]
    
    gnss_diff_monthly_all_16_23 = gnss_diff_monthly_23[gnss_diff_monthly_23['time'] >= pd.to_datetime('2016-07-15')]
    
    # Integrate to get cumulative displacement
    
    gnss_diff_daily_23['horiz_displ_filt_cum'] = gnss_diff_daily_23['horiz_displacement_filt'].cumsum()
    
    gnss_diff_monthly_all_16_23['horiz_displ_filt_cum'] = gnss_diff_monthly_all_16_23['horiz_displacement_filt'].cumsum()


    # Correct for rotational movement of the mast
    
    gnss_raw_incli['time'] = pd.to_datetime(gnss_raw_incli['time'])

    gnss_raw_incli.set_index('time', inplace = True, drop = False)

    gnss_raw_incli_daily = gnss_raw_incli.resample('D').mean()
    
    # Calculate net tilt using W and N components

    gnss_raw_incli_daily['tilt'] = np.sqrt(gnss_raw_incli_daily['inclinometer_west [°]']**2 + gnss_raw_incli_daily['inclinometer_west [°]']**2)

    # Subset to align time index to gnss_daily

    gnss_raw_incli_daily_16_23 = gnss_raw_incli_daily.loc["2016-07-01":"2023-08-26"]

    gnss_raw_incli_daily_16_23['date_only'] = gnss_raw_incli_daily_16_23['time'].dt.date

    # Calculate rotational displacement component

    gnss_raw_incli_daily_16_23['rotation_displ'] = 1.0 * np.sin(np.radians(gnss_raw_incli_daily_16_23['tilt'])) * 100 # to get cm

    # Correct due to offset in azimuth
    # Mast is tilting 86˚ away from the direction of RG flow

    gnss_raw_incli_daily_16_23['rotation_displ_corr'] = gnss_raw_incli_daily_16_23['rotation_displ']*np.cos(np.radians(274.5))

    # Resample to monthly

    gnss_raw_incli_monthly_16_23 = gnss_raw_incli_daily_16_23.resample('ME').last()

    gnss_diff_incli_monthly_16_23 = gnss_raw_incli_monthly_16_23.diff()

    gnss_diff_incli_monthly_16_23['date_only'] = gnss_diff_incli_monthly_16_23.index.date
    
    # Integrate inclination diff

    gnss_diff_daily_23['inclination_cum'] = gnss_diff_daily_23['inclination_filt'].cumsum()

    # Subtract rotational component to displacement data

    # First merge with inclination dataset

    gnss_diff_daily_23['date_only'] = gnss_diff_daily_23['time'].dt.date

    gnss_diff_daily_16_23_incli = pd.merge(gnss_diff_daily_23, 
                                           gnss_raw_incli_daily_16_23, 
                                           on='date_only', how='inner')  # use 'left', 'right', or 'outer' as needed

    gnss_diff_daily_16_23_incli['horiz_displ_filt_cum_corr'] = gnss_diff_daily_16_23_incli['horiz_displ_filt_cum']*100 - gnss_diff_daily_16_23_incli['rotation_displ']

    # Rotational component correction MONTHLY on cumulative displacement
    
    gnss_diff_monthly_all_16_23['date_only'] = gnss_diff_monthly_all_16_23['time'].dt.date

    gnss_diff_monthly_23_incli_merge = pd.merge(gnss_diff_monthly_all_16_23, 
                                           gnss_raw_incli_monthly_16_23, 
                                           on='date_only', how='inner')  # use 'left', 'right', or 'outer' as needed

    gnss_diff_monthly_23_incli_merge['horiz_displ_filt_cum_corr'] = gnss_diff_monthly_23_incli_merge['horiz_displ_filt_cum']*100 - gnss_diff_monthly_23_incli_merge['rotation_displ_corr']

    # Rotational component correction MONTHLY on velocity

    gnss_diff_monthly_23['date_only'] = gnss_diff_monthly_23['time'].dt.date

    gnss_vel_monthly_23_incli_merge = pd.merge(gnss_diff_monthly_23, 
                                           gnss_diff_incli_monthly_16_23, 
                                           on='date_only', how='inner')  # use 'left', 'right', or 'outer' as needed

    gnss_vel_monthly_23_incli_merge['horiz_displ_filt_corr'] = gnss_vel_monthly_23_incli_merge['horiz_displacement_filt']*100 - gnss_vel_monthly_23_incli_merge['rotation_displ_corr']


    # Get numeric columns to then do aggregation

    non_numeric_cols_gnss = gnss_diff_monthly_23.select_dtypes(exclude=['number']).columns

    gnss_diff_monthly_23_num = gnss_diff_monthly_23.drop(columns=non_numeric_cols_gnss)
    
    # Get yearly velocity
    
    gnss_diff_monthly_23_num['year'] = gnss_diff_monthly_23['time'].dt.year

    gnss_diff_yearly = gnss_diff_monthly_23_num.groupby('year').sum() # Annual total GNSS displacement
    gnss_diff_yearly.reset_index(inplace = True, drop = False)
    
    # Get annual statistics
    
    annual_means_gnss, annual_min_gnss, annual_max_gnss, min_timestamp_gnss, max_timestamp_gnss = aggregate_to_annual_from_daily(gnss_diff_daily_23, 
                                                                                                                                 sumSwitch = False,
                                                                                                                                 valid_days_switch = True)

    return {
        'gnss_diff_daily': gnss_diff_daily_23,
        'gnss_diff_monthly_23': gnss_diff_monthly_23,
        'gnss_diff_monthly_23_incli_merge': gnss_diff_monthly_23_incli_merge,
        'gnss_vel_monthly_23_incli_merge': gnss_vel_monthly_23_incli_merge,
        'gnss_diff_monthly_all_16_23': gnss_diff_monthly_all_16_23,
        'gnss_diff_yearly': gnss_diff_yearly,
        'annual_max_gnss': annual_max_gnss,
        }


def load_and_prepare_geodetic_data(data_dir = "geodetic_data"):
    
    # Data is at annual resolution
    
    geodetic_vel_data_path = os.path.join(data_dir, 'surface_vel_PERMOS_points.csv')
    
    geodetic_vel_data = pd.read_csv(geodetic_vel_data_path)
    
    geodetic_vel_data['date'] = pd.to_datetime(geodetic_vel_data['survey_date'])
    geodetic_vel_data['year'] = geodetic_vel_data['date'].dt.year

    # Subset to only take the reflector point on the same boulder as the GNSS station (COR_021)

    geodetic_vel_data_nearest = geodetic_vel_data[geodetic_vel_data['point_name'] == "COR_021"]
    
    geodetic_vel_data_nearest_17_23 = geodetic_vel_data_nearest[geodetic_vel_data_nearest['year'] > 2016]

    geodetic_vel_data_nearest_15_23 = geodetic_vel_data_nearest[geodetic_vel_data_nearest['year'] > 2014] # as 2016 year is missing

    # Prepare data to be ready to create step-line plot

    plot_geodetic_vel_point = expand_velocity_data(geodetic_vel_data_nearest_15_23)
    
    plot_geodetic_vel_point['date'] = pd.to_datetime(plot_geodetic_vel_point['date'])

    plot_geodetic_vel_point['decimal_year'] = plot_geodetic_vel_point['date'].dt.year + \
        (plot_geodetic_vel_point['date'].dt.day_of_year - 1) / \
        (plot_geodetic_vel_point['date'].dt.is_leap_year.replace({True: 366, False: 365}).infer_objects(copy=False)
         )

    return {
        'geodetic_vel_data': geodetic_vel_data,
        'plot_geodetic_vel_point': plot_geodetic_vel_point,
        'geodetic_vel_data_nearest_17_23': geodetic_vel_data_nearest_17_23
        }


def load_and_prepare_temp_data(data_dir = 'CORtemp_data'):
    
    temp_cor_15_path = os.path.join(data_dir, 'COR_0315_temp.csv')
    temp_cor_87_path = os.path.join(data_dir, 'COR_0287_temp.csv')
    
    temp_15_cor = pd.read_csv(temp_cor_15_path)
    temp_87_cor = pd.read_csv(temp_cor_87_path)
    
    temp_15_cor['time'] = pd.to_datetime(temp_15_cor['time'])
    temp_87_cor['time'] = pd.to_datetime(temp_87_cor['time'])
    
    # Interpolate gap-size specific approach
    
    temp_15_cor_interpolated = pd.DataFrame({
        col_name: selective_filling_interpolation(col_name, temp_15_cor[col_name])
        for col_name in temp_15_cor.columns
    })
    
    temp_87_cor_interpolated = pd.DataFrame({
        col_name: selective_filling_interpolation(col_name, temp_87_cor[col_name])
        for col_name in temp_87_cor.columns
    })

    
    # End 2015 borehole temp data to end of 2023

    temp_15_cor = temp_15_cor_interpolated[temp_15_cor_interpolated['time'].dt.year <= 2023]
    
    # Start 1987 borehole data from 1994 where continuous data starts:
        
    temp_87_cor = temp_87_cor_interpolated[temp_87_cor_interpolated['time'].dt.year >= 1994]
    
    # Create hydro_year column
    
    temp_15_cor['hydro_year'] = temp_15_cor['time'].apply(get_hydrological_year)

    temp_87_cor['hydro_year'] = temp_87_cor['time'].apply(get_hydrological_year)
    
    # Resample to annual stats

    temp_15_annual_means_sem, temp_15_annual_min, temp_15_annual_max, temp_15_annual_min_dates, temp_15_annual_max_dates = aggregate_to_annual_from_daily(temp_15_cor, sumSwitch = False,
                                                                                                                                                      valid_days_switch = True)
    
    temp_87_annual_means_sem, temp_87_annual_min, temp_87_annual_max, temp_87_annual_min_dates, temp_87_annual_max_dates = aggregate_to_annual_from_daily(temp_87_cor, sumSwitch = False,
                                                                                                                                                      valid_days_switch = True)

    # Clean the dataframe (if SEM values are not needed)
    
    sem_columns = [col for col in map(str, temp_15_annual_means_sem.columns) if 'sem' in col or 'sd' in col]
    temp_15_annual_means = temp_15_annual_means_sem.drop(columns = sem_columns)
    
    temp_15_annual_means.loc['2018','2.5'] = np.nan
    
    # List of thermistor depths as strings
    
    depth_str_15 = "'0.25', '0.5', '0.75', '1', '2', '3.01', '3.5', '4', '4.5', '5', '5.01', '6', '8', '10', '10.01', '12', '14', '16', '18', '20', '20.01', '25', '28', '30', '32', '34', '36', '38', '40', '42', '45', '50', '55', '60'"

    depth_str_15_all = "'0.25', '0.5', '0.75', '1', '1.5', '2', '2.5', '3', '3.01', '3.5', '4', '4.5', '5', '5.01', '6', '8', '10', '10.01', '12', '14', '16', '18', '20', '20.01', '25', '28', '30', '32', '34', '36', '38', '40', '42', '45', '50', '55', '58', '60'"

    # Get layer-specific mean temps
    
    # 2015
    
    temp_hole_15_AL_mean, temp_hole_15_AL_sd = get_layer_specific_mean(0.5, 2.5, temp_15_cor)

    temp_hole_15_core_mean, temp_hole_15_core_sd = get_layer_specific_mean(3.5, 25, temp_15_cor)

    temp_hole_15_shear_mean, temp_hole_15_shear_sd = get_layer_specific_mean(25, 28, temp_15_cor)

    temp_hole_15_layers = {
        'core': {'date': temp_15_cor['time'], 'mean': temp_hole_15_core_mean, 'sd': temp_hole_15_core_sd},
        'shear': {'date': temp_15_cor['time'], 'mean': temp_hole_15_shear_mean, 'sd': temp_hole_15_shear_sd},
        'AL': {'date': temp_15_cor['time'], 'mean': temp_hole_15_AL_mean, 'sd': temp_hole_15_AL_sd},
    }
    
    # 1987
    
    temp_hole_87_AL_mean, temp_hole_87_AL_sd = get_layer_specific_mean(0.5, 2.5, temp_87_cor)

    temp_hole_87_core_mean, temp_hole_87_core_sd = get_layer_specific_mean(3.5, 25, temp_87_cor)

    temp_hole_87_shear_mean, temp_hole_87_shear_sd = get_layer_specific_mean(25, 28, temp_87_cor)

    temp_hole_87_layers = {
        'core': {'date': temp_87_cor['time'], 'mean': temp_hole_87_core_mean, 'sd': temp_hole_87_core_sd},
        'shear': {'date': temp_87_cor['time'], 'mean': temp_hole_87_shear_mean, 'sd': temp_hole_87_shear_sd},
        'AL': {'date': temp_87_cor['time'], 'mean': temp_hole_87_AL_mean, 'sd': temp_hole_87_AL_sd},
    }
    
    # Extract AL only to a dataframe
    
    mean_temp_AL_df_15 = pd.DataFrame(temp_hole_15_layers['AL'])

    mean_temp_AL_df_87 = pd.DataFrame(temp_hole_87_layers['AL'])
    
    # Compute annual stats for AL mean temperature
    
    annual_means_temp_AL_15, annual_min_temp_AL_15, annual_max_temp_AL_15, min_timestamp_temp_AL_15, max_timestamp_temp_AL_15 = aggregate_to_annual_from_daily(mean_temp_AL_df_15, 
                                                                                                                                                sumSwitch = False,
                                                                                                                                                valid_days_switch = False)

    
    annual_means_temp_AL_87, annual_min_temp_AL_87, annual_max_temp_AL_87, min_timestamp_temp_AL_87, max_timestamp_temp_AL_87 = aggregate_to_annual_from_daily(mean_temp_AL_df_87, 
                                                                                                                                                sumSwitch = False,
                                                                                                                                                valid_days_switch = False)


    
    ## Extract ALT
    
    ALT_yearly_max_15_23, ALT_daily_16_23 = plot_contours_temp_ALT(temp_15_cor, depth_str_15, 
                                                        plot_switch = False)
    
    max_ALT_date = ALT_yearly_max_15_23['ALT_max_julian_day'].iloc[:-1]

    
    # Onset of freezing temperatures in 1987 and 2015 borehole
    
    freezing_onset_time_87 = find_freezing_start_temp(temp_87_cor, smoothSwitch = True)
    freezing_onset_time_15 = find_freezing_start_temp(temp_15_cor, smoothSwitch = True)

    # Zero curtain extraction - 2015
    
    ZCs_indices_15 = find_constant_periods(temp_15_cor['3.01'], step_tolerance = 0.01, period_tolerance = 0.01)
    spring_ZCs_times_15, autumn_ZCs_times_15 = get_start_end_duration(temp_15_cor, ZCs_indices_15)

    # Zero curtain extraction - 1987
    
    ZCs_indices_87 = find_constant_periods(temp_87_cor['2.55'], step_tolerance = 0.02, period_tolerance = 0.04)
    spring_ZCs_times_87, autumn_ZCs_times_87 = get_start_end_duration(temp_87_cor, ZCs_indices_87)

    # Manually remove 1994 and 2014 gap erroneous ZC:
    spring_ZCs_times_87 = spring_ZCs_times_87.drop(index = 0)
    autumn_ZCs_times_87 = autumn_ZCs_times_87.drop(index = 58)

    # Add rows for missing years
    all_years_87 = pd.DataFrame({'year': range(spring_ZCs_times_87['year'].min(), spring_ZCs_times_87['year'].max() + 1)})

    spring_ZCs_times_87_all = pd.merge(all_years_87, spring_ZCs_times_87, on = 'year', 
                                       how = 'left')

    # Convert dates to datetime format
    spring_ZCs_times_87['start_date'] = pd.to_datetime(spring_ZCs_times_87['start_date'])
    spring_ZCs_times_87['end_date'] = pd.to_datetime(spring_ZCs_times_87['end_date'])

    ## Ground heat flux
    
    # Q_G = k * (dT / dz)
    thermal_k = 2.5 # average for rock glacier material 

    # Find dz

    depth_list_15 = [float(x.strip("' ")) for x in depth_str_15_all.split(',')]

    dz_15 = [depth_list_15[i+1] - depth_list_15[i] for i in range(len(depth_list_15) - 1)]

    # Replace nans with -9999 to then filter out unrealistic dT/dz due to sensors data gaps

    temp_hole_15_nonan = temp_15_cor.fillna(-9999)

    # Find dT

    non_numeric_cols = temp_hole_15_nonan.select_dtypes(exclude=['number']).columns
    temp_hole_15_num = temp_hole_15_nonan.drop(columns=non_numeric_cols)
    temp_hole_15_num = temp_hole_15_num.drop(columns = ['hydro_year'])

    temp_15_dT = temp_hole_15_num.diff(axis = 1)

    # Remove first column as it is NaN
    temp_15_dT = temp_15_dT.drop('0.25', axis = 1)

    # Convert to ground heat flux

    # Ensure that the length of dz_15 matches the number of columns
    if len(dz_15) != len(temp_15_dT.columns):
        raise ValueError("Length of dz_15 must match the number of columns in temp_15_dT.")
        
    dz_15_series = pd.Series(dz_15, index=temp_15_dT.columns)

    ground_heat_flux_15 = temp_15_dT.div(dz_15_series, axis = 1) * thermal_k

    # Change values that are really negative (from -9999) or exactly 0 to NaN

    ground_heat_flux_15 = ground_heat_flux_15.replace({0: np.nan})

    ground_heat_flux_15[ground_heat_flux_15 < -8000] = np.nan

    ground_heat_flux_15[ground_heat_flux_15 > 10000] = np.nan

    # Flip sign to make more physical sense

    ground_heat_flux_15 = ground_heat_flux_15 * -1

    # Add time column

    ground_heat_flux_15['date'] = temp_15_cor['time']

    
    return {
        'temp_15_cor_daily': temp_15_cor,
        'temp_15_cor_annual': temp_15_annual_means,
        'temp_15_cor_layers': temp_hole_15_layers,
        'depths_15_cor': depth_str_15_all,
        'temp_15_annual_max': temp_15_annual_max,
        'temp_87_annual_max': temp_87_annual_max,
        'annual_max_temp_AL_15': annual_max_temp_AL_15,
        'annual_max_temp_AL_87': annual_max_temp_AL_87,
        'freezing_onset_time_87': freezing_onset_time_87,
        'freezing_onset_time_15': freezing_onset_time_15,
        'spring_ZCs_15': spring_ZCs_times_15,
        'spring_ZCs_87': spring_ZCs_times_87,
        "ground_heat_flux_15": ground_heat_flux_15,
        "max_ALT_date": max_ALT_date,
        }


def load_and_prepare_meteo_data(CORdata, data_dir = 'Meteo_data'):
    
    ## Import data
    
    permos_meteo_97_19_path = os.path.join(data_dir, 'murtel_level_2_hourly.csv')
    permos_meteo_19_23_path = os.path.join(data_dir, 'Corvatsch87_met.csv')
    
    # Meteo data at PERMOS station near borehole from 01/01/1997 to 31/03/2019
    permos_meteo_97_19 = pd.read_csv(permos_meteo_97_19_path)

    permos_meteo_97_19['date'] = pd.to_datetime(permos_meteo_97_19['date'])

    permos_meteo_97_19_useful = permos_meteo_97_19[['date', 'airtemp', 'surftemp', 'snowh', 'longout']]

    # Meteo data at PERMOS station until 2023
    permos_meteo_19_23 = pd.read_csv(permos_meteo_19_23_path, 
                               encoding='utf-8', 
                               parse_dates=['TimeStamp'], na_values=['NAN', 6999])

    permos_meteo_19_23.columns = ['date', 'airtemp', 'RH', 'VWND1', 'DWND1', 'VWND1_MAX', 'LWRup', 'LWRdown',
           'LWRnet', 'SWRup', 'SWRdown', 'SWRnet', 'snowh', 'surftemp', 'PLU_SUM10']

    # In 19-23 data the snowh is in cm instead of m so convert

    permos_meteo_19_23['snowh'] = permos_meteo_19_23['snowh']/100

    permos_meteo_19_23_useful = permos_meteo_19_23[['date', 'airtemp', 'surftemp', 'snowh', 'LWRdown']]
    permos_meteo_19_23_useful.rename(columns={'LWRdown': 'longout'}, inplace=True)
        
    # Combine all meteo data for 1997 - 2023 period

    permos_meteo_97_23 = pd.concat([permos_meteo_97_19_useful, permos_meteo_19_23_useful], ignore_index = True)
    permos_meteo_97_23['date'] = pd.to_datetime(permos_meteo_97_23['date'])

    permos_meteo_97_23 = permos_meteo_97_23[permos_meteo_97_23['date'] <= pd.to_datetime('2023-09-20')]
    
    daily_means = aggregate_to_daily(permos_meteo_97_23)
    
    
    ## PERMA-XT meteo data
    
    perma_xt_meteo_24_path = os.path.join(data_dir, 'permaxt_data_dom.csv')
    
    perma_xt_meteo_24 = pd.read_csv(perma_xt_meteo_24_path,
                                    parse_dates = ['TimeStamp'], na_values = 'NAN')

    perma_xt_meteo_24.rename(columns = {'TimeStamp' : 'date'}, inplace = True)

    daily_means_perma_xt_24 = aggregate_to_daily(perma_xt_meteo_24)
        
    # Taking daily sums for precip data
    
    # Set up indices so that they match and are date-only
    daily_means_perma_xt_24.set_index('date', drop = False, inplace = True)
    
    daily_means_perma_xt_24.index = pd.to_datetime(daily_means_perma_xt_24.index)
    
    daily_sums_perma_xt_24 = daily_means_perma_xt_24['Pluvio'].resample('D').sum()
    
    # Add column back to daily_means_perma_xt_24 
    daily_means_perma_xt_24['Pluvio_sum'] = (
        daily_means_perma_xt_24.index.floor('D').map(daily_sums_perma_xt_24)
    )
    
    ## Piz Corvatsch temp - import
    
    piz_temp_path = os.path.join(data_dir, 'corvatsch_temp.txt')

    piz_temp = pd.read_csv(piz_temp_path,
                           sep='\s+', skiprows = 8, encoding='latin1', 
                           low_memory=False)

    # Make date column
    piz_temp['date'] = pd.to_datetime(piz_temp[['JAHR', 'MO', 'TG']].astype(str).agg('-'.join, axis=1))
    piz_temp = piz_temp.drop(['STA', 'HH', 'MM'], axis = 1)
    piz_temp.rename(columns = {'211':'temp'}, inplace = True)
    piz_temp = piz_temp.drop(index=0)


    ## Piz Corvatsch precip - import
    
    piz_precip_path = os.path.join(data_dir, 'corvatsch_precip.txt')

    piz_precip = pd.read_csv(piz_precip_path, 
                             sep='\s+', skiprows = 8, encoding='latin1', 
                             low_memory=False)

    piz_precip.columns = ['STA', 'JAHR', 'MO', 'TG', 'HH', 'MM', 'precip']

    # Make date column
    piz_precip['date'] = pd.to_datetime(piz_precip[['JAHR', 'MO', 'TG']].astype(str).agg('-'.join, axis=1))
    piz_precip = piz_precip.drop(['STA', 'HH', 'MM'], axis = 1)

    piz_precip['date'] = pd.to_datetime(piz_precip['date'], errors='coerce')

    # Remove first fauly row
    piz_precip = piz_precip.drop(index=0)

    # Remove values unrealistically high
    piz_precip = piz_precip[piz_precip['precip'] <= 500]

    
    ## Create daily meteo means data frame with also missing dates
    
    # Add back missing rows from 2019-04-01 to 2019-09-30
    
    missing_dates = pd.date_range(start='2019-04-01', end='2019-09-30', freq='D')
    daily_means.index = pd.to_datetime(daily_means.index)
    daily_means_complete = daily_means.reindex(daily_means.index.union(missing_dates))

    # Daily means smoothening

    daily_means['airtemp_smooth'] = daily_means['airtemp'].rolling(30, center = True).mean()
    daily_means_complete['airtemp_smooth'] = daily_means_complete['airtemp'].rolling(30, center = True).mean()


    ## Fill long gaps based on data from Piz Corvatsch station
    
    # Make both index of both dataframes to date only
    
    daily_means_date_idx = daily_means.copy()
    
    daily_means_date_idx.set_index(daily_means_date_idx['date'].dt.date, 
                                   inplace = True, drop = False)
    
    piz_temp.set_index(piz_temp['date'].dt.date, inplace = True, drop = False)
    
    # Merge dataframes
    
    piz_permos_temp = pd.merge(daily_means_date_idx, piz_temp, left_index = True, right_index = True)
    
    piz_permos_temp.dropna(subset = ['airtemp', 'temp'], inplace = True)
    
    # Regression 
    
    slope_piz_permos_T, intercept_piz_permos_T, r_value_piz_permos_T, p_value_piz_permos_T, std_err_piz_permos_T = linregress(piz_permos_temp['temp'], 
                                                                                                piz_permos_temp['airtemp'])
    r2_piz_permos_T = r_value_piz_permos_T**2
    print("R^2 between in-situ air temp and Piz Corvatsch air temp:", r2_piz_permos_T)
    
    # Add in missing rows for data gaps in daily_means
    
    full_range_dates_daily_means = pd.date_range(start=piz_permos_temp.index.min(), 
                                                 end=piz_permos_temp.index.max(), 
                                                 freq='D')
    
    daily_means_date_idx_complete = piz_permos_temp.reindex(full_range_dates_daily_means)
    
    # Function to convert from corv_temp to permos_temp
    
    def fill_permos_temp_gap(corv_temp):
    
        temp_interpolated = intercept_piz_permos_T + slope_piz_permos_T * corv_temp
    
        return pd.Series(temp_interpolated)    
    
    # Get a long series of PERMOS temperatures converted from Piz Corvatsch which runs for the whole time period
    
    permos_temp_interpolated_from_piz = pd.DataFrame([fill_permos_temp_gap(x) for x in piz_temp['temp']])
    permos_temp_interpolated_from_piz.index = piz_temp.index 
    
    # Find dates where the PERMOS temp is missing
    
    missing_dates_with_no_temp = daily_means_date_idx_complete[daily_means_date_idx_complete['airtemp'].isna()].index
    
    # Fill the temp values for those dates using the converted temps from Piz Corvatsch
    
    permos_temp_interpolated_from_piz.index = pd.to_datetime(permos_temp_interpolated_from_piz.index)
    
    print("permos_temp_interpolated_from_piz.index", permos_temp_interpolated_from_piz.index[0:1])
    print("missing_dates_with_no_temp.index", missing_dates_with_no_temp[0:1])
    
    # Find which of the missing dates are available in permos_temp_interpolated_from_piz
    
    matching_dates = missing_dates_with_no_temp.intersection(permos_temp_interpolated_from_piz.index)
    
    print("Matching dates found in permos_temp_interpolated_from_piz:", len(matching_dates))
    
    permos_temp_interpolated_from_piz = pd.Series(permos_temp_interpolated_from_piz[0])
    
    daily_means_date_idx_complete.loc[missing_dates_with_no_temp, 'airtemp'] = (
        daily_means_date_idx_complete.loc[missing_dates_with_no_temp, 'airtemp']
        .combine_first(permos_temp_interpolated_from_piz.loc[missing_dates_with_no_temp])
    )
    
    ## Extend daily_means_date_idx_complete to 2023-09-19
    
    # Select the part of daily_means that comes after 2020-01-04 
    # (that's when the Piz Corvatsch data ends)
    
    daily_means_after_2020 = daily_means[daily_means['date'] > '2020-01-04']
    
    # Concatenate the two DataFrames
    
    daily_means_extended = pd.concat([daily_means_date_idx_complete, daily_means_after_2020])
    
    # Ensure the index remains in datetime format
    
    daily_means_extended = daily_means_extended.sort_index()
    
    # Get also a smooth version of the new complete airtemp
    
    daily_means_extended['airtemp_smooth'] = daily_means_extended['airtemp'].rolling(window = 10).mean()
    
    daily_means_extended['date'] = daily_means_extended.index
    
    daily_means_extended.drop(['date_x', 'date_y'],axis = 1, inplace = True)
    
    # Fill in missing snow data
    
    daily_means_extended['month_day'] = daily_means_extended['date'].dt.strftime('%m-%d')

    # Identify the missing date range
    missing_dates = pd.date_range(start='2019-03-31', end='2019-09-30')

    # Compute average snow height for each month-day across all years (excluding 2019)
    avg_snow_height = (
        daily_means_extended[~daily_means_extended['date'].dt.year.eq(2019)]
        .groupby('month_day')['snowh']
        .mean()
    )

    # Create a DataFrame for the missing dates
    missing_snow_df = pd.DataFrame({'date': missing_dates})
    missing_snow_df['month_day'] = missing_snow_df['date'].dt.strftime('%m-%d')

    # Fill missing values using the computed averages
    missing_snow_df['snowh'] = missing_snow_df['month_day'].map(avg_snow_height)

    #daily_means_extended['snowh'].fillna(missing_snow_df['snowh'])

    # Merge the filled values back into the original dataset
    daily_means_extended = pd.merge(daily_means_extended, 
                                         missing_snow_df, on='date', how='left',
                                         suffixes=("","_new"))

    daily_means_extended['snowh'] = daily_means_extended['snowh'].fillna(daily_means_extended['snowh_new'])

    daily_means_extended.drop(columns = ['snowh_new','month_day_new'],
                                   inplace = True)

    daily_means_extended.set_index('date', inplace = True, drop = False)
    
    
    ## Get early winter means
    
    # Mean early snow accumulation (Nov-Jan)

    early_winter_daily, early_winter_means = early_winter_annual_mean(daily_means)
    
    
    ## Snow end timing:
        
    # Reset the daily_means index to do operations such as finidng the previous row in iterrows

    daily_means_melt = daily_means[(daily_means.index.month >= 4) & (daily_means.index.month <= 8)]

    snow_end_time = get_snow_end_time(daily_means_melt)

    snow_end_time['julian_day'] = snow_end_time['LastSnowDate'].apply(get_julian_day)
    
    snow_end_time_16 = snow_end_time[snow_end_time['HydroYear'] >= 2015]

    snow_end_time_16['year'] = snow_end_time_16['HydroYear'] + 1

    snow_end_time_16['julian_day'] = snow_end_time_16['LastSnowDate'].apply(get_julian_day)

    
    ## Rainfall data processing
    
    # Extract Piz Corvatsch warm phase precipitation
    
    piz_precip_phases, piz_precip_warm, piz_precip_cold = get_phase_out_of_date(piz_precip, CORdata['freezing_onset_time_87'], snow_end_time)
    
    # Prepare to correlate between PERMA-XT rainfall and Piz Corvatsch rainfall
    
    # Subset the two dataframes such that they cover the exact same date range
    
    start_date_perma_xt_precip = pd.to_datetime(perma_xt_meteo_24['date'][0].date())

    end_date_piz_precip = pd.to_datetime(piz_precip['date'].iloc[-1].date())

    piz_precip_20_24 = piz_precip[(piz_precip['date'] >= start_date_perma_xt_precip) &
                                  (piz_precip['date'] <= pd.to_datetime("2023-12-24"))]

    daily_means_perma_xt_20_24 = daily_means_perma_xt_24[daily_means_perma_xt_24['date'] <= end_date_piz_precip]
    
    # Reset the indices so that they are not in datetime
    
    daily_means_perma_xt_20_24 = daily_means_perma_xt_20_24.reset_index(drop = True)
    piz_precip_20_24 = piz_precip_20_24.reset_index(drop = True)

    # Merge the PERMA-XT and Piz Corvatsch precip together

    daily_means_perma_xt_20_24['date'] = pd.to_datetime(daily_means_perma_xt_20_24['date']).dt.date
    piz_precip_20_24['date'] = pd.to_datetime(piz_precip_20_24['date']).dt.date
    
    print("daily means PERMA-XT index:", daily_means_perma_xt_20_24.index)
    print('piz_precip_20_24 index:', piz_precip_20_24.index)
    
    piz_perma_merged_precip_20_24 = pd.merge(daily_means_perma_xt_20_24, 
                                             piz_precip_20_24, on = 'date',
                                             how='inner')

    piz_perma_merged_precip_20_24 = piz_perma_merged_precip_20_24[['date', 'Pluvio_sum', 'precip']]

    piz_perma_merged_precip_20_24.rename(columns={'Pluvio_sum': 'precip_perma_xt', 'precip': 'precip_piz'}, inplace=True)
    
    print("Precip merged dataframe:", piz_perma_merged_precip_20_24.head())

    # Extract only the warm phase precipitation
    
    piz_perma_precip_20_24_phases, piz_perma_precip_20_24_warm, piz_perma_precip_20_24_cold = get_phase_out_of_date(piz_perma_merged_precip_20_24, 
                                                                                                                    CORdata['freezing_onset_time_15'], 
                                                                                                                    snow_end_time_16)

    piz_perma_precip_20_24_warm['year'] = piz_perma_precip_20_24_warm['date'].dt.year
    

    ## Correlate Piz Corvatsch rainfall with in-situ PERMA-XT rainfall
    
    slope_precip, intercept_precip, r_value_precip, p_value_precip, std_err_precip = linregress(piz_perma_precip_20_24_warm['precip_piz'], 
                                                                                                piz_perma_precip_20_24_warm['precip_perma_xt'])

    # Extrapolate PERMA-XT series using Piz Corvatsch rainfall as an input and their regression
    
    def extrapolate_precip(corv_precip):

        precip_extrapolated = intercept_precip + slope_precip * corv_precip

        return pd.Series(precip_extrapolated)    
    
    perma_xt_warm_precip_extrapolated = pd.DataFrame([extrapolate_precip(x) for x in piz_precip_warm['precip']])


    # Add back time column
    
    piz_precip_dates_warm = piz_precip_warm['date']
    piz_precip_dates_warm.reset_index(drop = True, inplace = True)

    perma_xt_warm_precip_extrapolated_times = pd.merge(perma_xt_warm_precip_extrapolated, 
                                                  piz_precip_dates_warm, left_index = True,
                                                  right_index = True)

    # Rename precip column

    perma_xt_warm_precip_extrapolated_times.rename(columns = {0:'precip'}, inplace = True)
    
    
    ## Snow melt energy
    
    # Only take snow to be melting when airtemp is > -3˚C
    daily_means_melt = daily_means_extended[daily_means_extended['airtemp'] > -3]

    # Try smoothening snowh first
    daily_means_melt['snowh_smooth'] = daily_means_melt['snowh'].rolling(window = 10).mean()

    daily_means_melt['snowmelt'] = daily_means_melt['snowh_smooth'].diff()

    # only consider snow height loss and make any increase of snow h to 0 since there is no melt
    daily_means_melt['snowmelt'] = daily_means_melt['snowmelt'].mask(daily_means_melt['snowmelt'] > 0 , 0)

    # Calculate energy
    snow_density = 300 # kg/m^3
    L_f = 334e3 # J/kg

    daily_means_melt['melt_nrg'] = ( abs(daily_means_melt['snowmelt']) * snow_density * L_f ) / (24*60*60) # W/m^2

    # Remove outliers due to data gaps which skips rows of missing dates so big jump in snowh causes very large Q_m

    daily_means_melt = daily_means_melt[daily_means_melt['melt_nrg'] < 147]

    # Aggregate to monthly

    daily_means_melt_clean = daily_means_melt.filter(items = ['date','melt_nrg'])
    
    snowmelt_annual_means, snowmelt_annual_min, snowmelt_annual_max, snowmelt_min_time, snowmelt_max_time = aggregate_to_annual_from_daily(daily_means_melt_clean, sumSwitch = False,
                                                                                                                                                     valid_days_switch = False)
    snowmelt_max_time['julian_day'] = snowmelt_max_time['melt_nrg'].apply(get_julian_day)

    
    return {
        'daily_means': daily_means,
        'daily_means_extended': daily_means_extended,
        'early_winter_daily': early_winter_daily,
        'rainfall_extrapolated': perma_xt_warm_precip_extrapolated_times,
        'snow_end_time_16': snow_end_time_16,
        'snowmelt_nrg': daily_means_melt_clean,
        'snowmelt_max_time': snowmelt_max_time,
        }

def prepare_corr_data(GNSS_data, temp_data, meteo_data):
    
    # Extract relevant variables
    
    annual_max_AL_temp_87 = temp_data['annual_max_temp_AL_87']['mean'].loc['1997':'2022']
    annual_max_AL_temp_15 = temp_data['annual_max_temp_AL_15']['mean'].loc[:'2022']
    
    spring_ZCs_15_end_dates = temp_data['spring_ZCs_15']['end_date'].iloc[:-1].apply(get_julian_day)
    spring_ZCs_87_end_dates = temp_data['spring_ZCs_87']['end_date'].apply(get_julian_day)
    
    max_ALT_date = temp_data['max_ALT_date']
    
    snowmelt_max_time_16_22 = meteo_data['snowmelt_max_time'].loc['2016':'2022', 'julian_day']
    
    annual_max_gnss_displacement = GNSS_data['annual_max_gnss']['horiz_displacement_filt']

    
    # Set all indices the same as the GNSS
    
    max_ALT_date.index = annual_max_gnss_displacement.index
    
    # Perform linear regressions
    
    slope_end_SZC_ALT_max_date, intercept_end_SZC_ALT_max_date, r_value_end_SZC_ALT_max_date, p_value_end_SZC_ALT_max_date, std_err_end_SZC_ALT_max_date = linregress(spring_ZCs_15_end_dates, 
                                                                                                                                        max_ALT_date)

    r2_end_SZC_ALT_max_date = round((r_value_end_SZC_ALT_max_date**2), 2)

    # Merge between SZC end dates and max AL temp to only get years with both values
    
    spring_ZCs_87_end_dates.index = temp_data['spring_ZCs_87']['year']
    
    spring_ZCs_87_end_dates.index = pd.PeriodIndex(
        spring_ZCs_87_end_dates.index.astype(str), 
        freq="Y"
    )    
    
    end_SZC_AL_T87_df = pd.merge(spring_ZCs_87_end_dates, annual_max_AL_temp_87, left_index=True, right_index=True)

    end_SZC_AL_T87_df_clean = end_SZC_AL_T87_df.dropna()
    
    print('spring_ZCs_87_end_dates index:', spring_ZCs_87_end_dates.index)
    print('annual_max_AL_temp_87 index:', annual_max_AL_temp_87.index)


    spring_ZC_end_dates_87_clean = end_SZC_AL_T87_df_clean.iloc[:,0]
    annual_max_AL_temp_87_clean = end_SZC_AL_T87_df_clean.iloc[:,1]

    slope_end_SZC_AL_T87, intercept_end_SZC_AL_T87, r_value_end_SZC_AL_T87, p_value_end_SZC_AL_T87, std_err_end_SZC_AL_T87 = linregress(spring_ZC_end_dates_87_clean, 
                                                                                                                                        annual_max_AL_temp_87_clean)

    r2_end_SZC_AL_T87 = round((r_value_end_SZC_AL_T87**2), 2)


    # Second row

    slope_end_SZC_GNSS, intercept_end_SZC_GNSS, r_value_end_SZC_GNSS, p_value_end_SZC_GNSS, std_err_end_SZC_GNSS = linregress(spring_ZCs_15_end_dates, 
                                                                                                          annual_max_gnss_displacement*1000)

    r2_end_SZC_GNSS = round((r_value_end_SZC_GNSS**2), 2)


    slope_AL_temp_GNSS, intercept_AL_temp_GNSS, r_value_AL_temp_GNSS, p_value_AL_temp_GNSS, std_err_AL_temp_GNSS = linregress(annual_max_AL_temp_15, 
                                                                                                                                        annual_max_gnss_displacement*1000)

    r2_AL_temp_GNSS = round((r_value_AL_temp_GNSS**2), 2)



    slope_ALT_max_date_GNSS, intercept_ALT_max_date_GNSS, r_value_ALT_max_date_GNSS, p_value_ALT_max_date_GNSS, std_err_ALT_max_date_GNSS = linregress(max_ALT_date, 
                                                                                                                                        annual_max_gnss_displacement*1000)

    r2_ALT_max_date_GNSS = round((r_value_ALT_max_date_GNSS**2), 2)


    slope_Qm_maxtime_GNSS, intercept_Qm_maxtime_GNSS, r_value_Qm_maxtime_GNSS, p_value_Qm_maxtime_GNSS, std_err_Qm_maxtime_GNSS = linregress(snowmelt_max_time_16_22, 
                                                                                                                                         annual_max_gnss_displacement*1000)

    r2_Qm_maxtime_GNSS = round((r_value_Qm_maxtime_GNSS**2), 2)

    return {
     # --- Data returned ---
     'annual_max_AL_temp_87': annual_max_AL_temp_87,
     'annual_max_AL_temp_15': annual_max_AL_temp_15,
     'spring_ZCs_15_end_dates': spring_ZCs_15_end_dates,
     'spring_ZCs_87_end_dates': spring_ZCs_87_end_dates,
     'max_ALT_date': max_ALT_date,
     'snowmelt_max_time_16_22': snowmelt_max_time_16_22,
     'annual_max_gnss_displacement': annual_max_gnss_displacement,
 
     # --- R² values ---
     'r2_end_SZC_ALT_max_date': r2_end_SZC_ALT_max_date,
     'r2_end_SZC_AL_T87': r2_end_SZC_AL_T87,
     'r2_end_SZC_GNSS': r2_end_SZC_GNSS,
     'r2_AL_temp_GNSS': r2_AL_temp_GNSS,
     'r2_ALT_max_date_GNSS': r2_ALT_max_date_GNSS,
     'r2_Qm_maxtime_GNSS': r2_Qm_maxtime_GNSS,
 
     # --- Full regression output ---
     'end_SZC_ALT_max_date_reg': {
         'slope': slope_end_SZC_ALT_max_date,
         'intercept': intercept_end_SZC_ALT_max_date,
         'p_value': p_value_end_SZC_ALT_max_date,
         'std_err': std_err_end_SZC_ALT_max_date
     },
 
     'end_SZC_AL_T87_reg': {
         'slope': slope_end_SZC_AL_T87,
         'intercept': intercept_end_SZC_AL_T87,
         'p_value': p_value_end_SZC_AL_T87,
         'std_err': std_err_end_SZC_AL_T87
     },
 
     'end_SZC_GNSS_reg': {
         'slope': slope_end_SZC_GNSS,
         'intercept': intercept_end_SZC_GNSS,
         'p_value': p_value_end_SZC_GNSS,
         'std_err': std_err_end_SZC_GNSS
     },
 
     'AL_temp_GNSS_reg': {
         'slope': slope_AL_temp_GNSS,
         'intercept': intercept_AL_temp_GNSS,
         'p_value': p_value_AL_temp_GNSS,
         'std_err': std_err_AL_temp_GNSS
     },
 
     'ALT_max_date_GNSS_reg': {
         'slope': slope_ALT_max_date_GNSS,
         'intercept': intercept_ALT_max_date_GNSS,
         'p_value': p_value_ALT_max_date_GNSS,
         'std_err': std_err_ALT_max_date_GNSS
     },
 
     'Qm_maxtime_GNSS_reg': {
         'slope': slope_Qm_maxtime_GNSS,
         'intercept': intercept_Qm_maxtime_GNSS,
         'p_value': p_value_Qm_maxtime_GNSS,
         'std_err': std_err_Qm_maxtime_GNSS
     }}
 
