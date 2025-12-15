#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:32:29 2025

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

# DEFINE FUNCTIONS USED TO PROCESS INCLINOMETER DATA

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


# Get hydrological year starting in Sept 1
def get_hydrological_year(date):
    return date.year if date.month >= 9 else date.year - 1

# -------------- Import data --------------------- #


def load_and_prepare_saa_data(data_dir="SAA_data"):
    """Load, clean, and prepare SAA data for analysis and plotting."""

    # ---------------------------
    # Load .mat files
    # ---------------------------
    SAA_16_17_path = os.path.join(data_dir, "multi_saa_allcart_16_17.mat")
    SAA_17_18_path = os.path.join(data_dir, "multi_saa_allcart_17_18.mat")
    SAA_18_23_path = os.path.join(data_dir, "multi_saa_allcart_18_23.mat")

    SAA_16_17 = scipy.io.loadmat(SAA_16_17_path)
    SAA_17_18 = scipy.io.loadmat(SAA_17_18_path)
    SAA_18_23 = scipy.io.loadmat(SAA_18_23_path)

    # ---------------------------
    # Extract variables
    # ---------------------------
    SAA_16_17_x, SAA_16_17_y, SAA_16_17_z, SAA_16_17_time, SAA_16_17_temp = extract_vars_mat(SAA_16_17)
    SAA_17_18_x, SAA_17_18_y, SAA_17_18_z, SAA_17_18_time, SAA_17_18_temp = extract_vars_mat(SAA_17_18)
    SAA_18_23_x, SAA_18_23_y, SAA_18_23_z, SAA_18_23_time, SAA_18_23_temp = extract_vars_mat(SAA_18_23)

    # ---------------------------
    # Combine data
    # ---------------------------
    SAA_16_18_x = pd.concat([pd.DataFrame(SAA_16_17_x), pd.DataFrame(SAA_17_18_x)], ignore_index=True)
    SAA_16_18_y = pd.concat([pd.DataFrame(SAA_16_17_y), pd.DataFrame(SAA_17_18_y)], ignore_index=True)
    SAA_16_18_z = pd.concat([pd.DataFrame(SAA_16_17_z), pd.DataFrame(SAA_17_18_z)], ignore_index=True)
    SAA_16_18_time = pd.concat([pd.DataFrame(SAA_16_17_time), pd.DataFrame(SAA_17_18_time)], ignore_index=True)
    SAA_16_18_temp = pd.concat([pd.DataFrame(SAA_16_17_temp), pd.DataFrame(SAA_17_18_temp)], ignore_index=True)

    SAA_16_24_x = pd.concat([SAA_16_18_x, pd.DataFrame(SAA_18_23_x)], ignore_index=True)
    SAA_16_24_y = pd.concat([SAA_16_18_y, pd.DataFrame(SAA_18_23_y)], ignore_index=True)
    SAA_16_24_z = pd.concat([SAA_16_18_z, pd.DataFrame(SAA_18_23_z)], ignore_index=True)
    SAA_16_24_time = pd.concat([SAA_16_18_time, pd.DataFrame(SAA_18_23_time)], ignore_index=True)
    SAA_16_24_temp = pd.concat([SAA_16_18_temp, pd.DataFrame(SAA_18_23_temp)], ignore_index=True)

    # ---------------------------
    # Derived data
    # ---------------------------
    mean_depths = abs(SAA_18_23_z.mean(axis=0))
    mean_depths_pd = pd.DataFrame(mean_depths)

    SAA_16_24_x1, SAA_16_24_y1, SAA_16_24_z1 = subtract_first_value(
        SAA_16_24_x, SAA_16_24_y, SAA_16_24_z, SAA_16_24_time,
        start_date="2016-01-06 12:05:00"
    )

    SAA_16_24_tot = subtract_first_value_get_tot(
        SAA_16_24_x, SAA_16_24_y, SAA_16_24_time,
        start_date="2016-01-06 12:05:00",
        one_year_switch=False,
        smoothSwitch=True,
        dailySwitch=True
    )

    # Clip after 2016-04-01 (settling)
    if 'time' not in SAA_16_24_tot.columns:
        SAA_16_24_tot = SAA_16_24_tot[SAA_16_24_tot.index > pd.to_datetime("2016-04-01 00:00:00")]
    else:
        SAA_16_24_tot = SAA_16_24_tot[SAA_16_24_tot['time'] > pd.to_datetime("2016-04-01 00:00:00")]

    # ---------------------------
    # Clip before rockfall (2023-09-15 cutoff)
    # ---------------------------
    cutoff_date = pd.to_datetime("2023-09-15 00:00:00")

    SAA_16_23_x1 = SAA_16_24_x1[SAA_16_24_x1['time'] < cutoff_date]
    SAA_16_23_y1 = SAA_16_24_y1[SAA_16_24_y1['time'] < cutoff_date]
    SAA_16_23_z1 = SAA_16_24_z1[SAA_16_24_z1['time'] < cutoff_date]

    if 'time' not in SAA_16_24_tot.columns:
        SAA_16_23_tot = SAA_16_24_tot[SAA_16_24_tot.index < cutoff_date]
    else:
        SAA_16_23_tot = SAA_16_24_tot[SAA_16_24_tot['time'] < cutoff_date]

    # ---------------------------
    # Daily & monthly depth values
    # ---------------------------
    # Merge z and time
    SAA_16_24_z_merge = pd.merge(SAA_16_24_z, SAA_16_24_time, left_index=True, right_index=True)
    SAA_16_24_z_merge.set_index('0_y', inplace=True, drop=True)

    # Daily resample
    SAA_16_24_z_daily = SAA_16_24_z_merge.resample("D").mean()
    SAA_16_24_z_daily = SAA_16_24_z_daily[SAA_16_24_z_daily.index >= pd.to_datetime("2016-04-01 00:00:00")]

    # Clip to before rockfall
    SAA_16_23_z_daily = SAA_16_24_z_daily[SAA_16_24_z_daily.index < cutoff_date]

    # Monthly resample
    SAA_16_23_z_monthly = SAA_16_23_z_daily.resample('ME').mean()
    SAA_16_23_z_monthly = SAA_16_23_z_monthly.iloc[:-1]  # drop last row

    # ---------------------------
    # Return everything neatly
    # ---------------------------
    return {
        # Full dataset
        "x": SAA_16_24_x,
        "y": SAA_16_24_y,
        "z": SAA_16_24_z,
        "time": SAA_16_24_time,
        "temp": SAA_16_24_temp,
        "mean_depths": mean_depths_pd,
        "x1_24": SAA_16_24_x1,
        "y1_24": SAA_16_24_y1,
        "z1_24": SAA_16_24_z1,
        "total_deformation_24": SAA_16_24_tot,

        # Clipped dataset (before rockfall)
        "x1_23": SAA_16_23_x1,
        "y1_23": SAA_16_23_y1,
        "z1_23": SAA_16_23_z1,
        "total_deformation_23": SAA_16_23_tot,

        # Daily & monthly
        "z_daily": SAA_16_24_z_daily,
        "z_daily_23": SAA_16_23_z_daily,
        "z_monthly_23": SAA_16_23_z_monthly,
    }


# Debug run
if __name__ == "__main__":
    data = load_and_prepare_saa_data()
    print("Available keys:", list(data.keys()))
    print(data["z_monthly_23"].head())


def compute_velocities(data_dict):
    
    """Convert to monthly resolution of displacement. Convert to velocity 
    at different temporal resolutions. """

    # Make sure index is in correct format

    if not isinstance(data_dict['total_deformation_23'].index, pd.DatetimeIndex):
        
        if 'time' in data_dict['total_deformation_23'].columns:
            
            data_dict['total_deformation_23']['time'] = pd.to_datetime(data_dict['total_deformation_23']['time'], errors='coerce')  # Convert to datetime, handling errors
            
            if data_dict['total_deformation_23']['time'].isna().any():
                print("Warning: Some entries in the 'time' column could not be converted to datetime.")
                
            data_dict['total_deformation_23'] = data_dict['total_deformation_23'].set_index('time')  # Set 'time' column as the index
        else:
            print("Error: 'time' column not found in the DataFrame.")
            
    SAA_16_23_tot_monthly = data_dict['total_deformation_23'].resample('ME').last()

    # Remove last row which only has some data from Sept 2023 so sum is not complete

    SAA_16_23_tot_monthly = SAA_16_23_tot_monthly.iloc[:-1]

    # Remove time column

    SAA_16_23_tot_monthly.drop(columns = 'time', inplace = True)

    # Yearly deformation values (last value from each year)

    SAA_16_23_tot_yearly = data_dict['total_deformation_23'].resample('YE').last()

    # Hydro year deformation values (end of hydro year here defined as 31.08)

    data_dict['total_deformation_23'].loc[:,'hydro_year'] = (
        data_dict['total_deformation_23']['time'].apply(get_hydrological_year))

    SAA_16_23_tot_hydro_yearly = data_dict['total_deformation_23'].groupby('hydro_year').last()

    # Remove last row which only takes days from 2023-08-31 to last data day of 2023-09-14

    SAA_16_23_tot_hydro_yearly = SAA_16_23_tot_hydro_yearly.iloc[:-1]


    # Convert to VELOCITY

    # Daily

    non_numeric_cols = data_dict['total_deformation_23'].select_dtypes(exclude=['number']).columns

    SAA_16_23_tot_num = data_dict['total_deformation_23'].drop(columns = non_numeric_cols)

    SAA_vel_ext_daily = SAA_16_23_tot_num.diff()

    # Smoothen

    SAA_vel_ext_daily_smooth = SAA_vel_ext_daily.rolling(window = 30, center = True).mean()

    # Monthly

    SAA_vel_ext_monthly = SAA_16_23_tot_monthly.diff()
    
    SAA_vel_ext_monthly_16 = SAA_vel_ext_monthly.loc[pd.to_datetime('2016-06-30'):]
    SAA_vel_ext_monthly_16['surf_cum_defo'] = SAA_vel_ext_monthly_16[80].cumsum()

    # Annual

    SAA_vel_yearly = SAA_16_23_tot_yearly.diff()

    # Velocity per hydro year

    # Finding difference to get a displacement per year at all depths

    non_numeric_cols_SAA = SAA_16_23_tot_hydro_yearly.select_dtypes(exclude=['number']).columns

    SAA_16_23_tot_hydro_yearly_num = SAA_16_23_tot_hydro_yearly.drop(columns=non_numeric_cols_SAA)

    SAA_16_23_vel_hydro_yearly = SAA_16_23_tot_hydro_yearly_num.diff()

    # Remove first row as there is no difference since there is no previous year

    SAA_16_23_vel_hydro_yearly['time'] = SAA_16_23_tot_hydro_yearly['time']

    SAA_16_23_vel_hydro_yearly = SAA_16_23_vel_hydro_yearly.iloc[1:]

    # Find average displacement per year for each depth across the whole period

    SAA_vel_hydro_yearly_means = SAA_16_23_vel_hydro_yearly.mean()

    SAA_vel_hydro_yearly_means = SAA_vel_hydro_yearly_means[:-1]

    SAA_vel_hydro_yearly_means = SAA_vel_hydro_yearly_means.astype(float)
    return {
        "tot_monthly": SAA_16_23_tot_monthly,
        "tot_monthly_16": SAA_vel_ext_monthly_16,
        "tot_yearly": SAA_16_23_tot_yearly,
        "tot_hydro_yearly": SAA_16_23_tot_hydro_yearly,
        "vel_daily": SAA_vel_ext_daily,
        "vel_daily_smooth": SAA_vel_ext_daily_smooth,
        "vel_monthly": SAA_vel_ext_monthly,
        "vel_yearly": SAA_vel_yearly,
        "vel_hydro_yearly": SAA_16_23_vel_hydro_yearly,
        "vel_hydro_yearly_means": SAA_vel_hydro_yearly_means,
    }


def compute_layer_deformation(data_dict, vel_data):
    
    """ Calculate layer-specific velocity at a daily, annual and total mean
    resolution. Calculate strain rate, layer-specific strain rate
    """
    
    # Layer-specific annual mean velocity

    # AL = velocity at 0 m - velocity at 3.5 m
    SAA_vel_hydro_yearly_mean_AL = vel_data['vel_hydro_yearly_means'][80] - vel_data['vel_hydro_yearly_means'][73]

    # Ice core = velocity at 3.5 m - velocity at 25.5 m
    SAA_vel_hydro_yearly_mean_core = vel_data['vel_hydro_yearly_means'][73] - vel_data['vel_hydro_yearly_means'][29]

    # Shear zone = velocity at 25.5 m - velocity at 29 m
    SAA_vel_hydro_yearly_mean_shear = vel_data['vel_hydro_yearly_means'][29] - vel_data['vel_hydro_yearly_means'][22]


    # VERTICAL STRAIN RATE (dv/dz)

    #SAA_vel_ext_monthly_num = SAA_vel_ext_monthly.drop(columns = 'time')

    SAA_dvdz_ext_monthly = vel_data['vel_monthly'].diff(axis = 1)

    SAA_dvdz_ext_monthly = SAA_dvdz_ext_monthly / 500 # 50 cm vertical spacing between sensors

    # Extract times (important for later plotting vertical strain rate profiles)

    SAA_dvdz_monthly_times = pd.Series(SAA_dvdz_ext_monthly.index)

    # Layer-specific annual mean strain rates

    # AL = diff. in velocity from top to bottom (SAA_vel_hydro_yearly_mean_AL) / 3500 mm
    SAA_dvdz_AL_annual_mean = SAA_vel_hydro_yearly_mean_AL / 3500


    # Ice core height in mm = 22000 (3.5 m to 25.5 m)
    SAA_dvdz_core_annual_mean = SAA_vel_hydro_yearly_mean_core / 22000

    # Shear zone height in mm = 1500 (26 to 28.5 m)
    SAA_dvdz_shear_annual_mean = SAA_vel_hydro_yearly_mean_shear / 1500


    # DISPLACEMENT for each LAYER DAILY

    # Subtract the deformation at the top of the layer by the deformation at the bottom of the layer

    # The ALT is taken to be 3.5 m until 2021 after which it increases to 4.5 m
    # The shear zone extent is taken to be from 25 to 29 m

    defo_shearzone_daily = []
    defo_core_daily = []
    defo_AL_daily = []
    months = []
    hydro_years = []

    # Iterate for every day in this year
    for index, row in vel_data['vel_daily'].iterrows():  
        
        date = row.name # get the value of the index (in this case a date)
        year = date.year
        month = date.month
        
        hydro_year = year if month > 8 else year - 1
        
        defo_at_0m = row[80] 
        defo_at_3_5_m = row[73] 
        defo_at_4_5_m = row[71]
        defo_at_25m = row[30]
        defo_at_29m = row[22] # can vary this
        
        defo_shearzone = defo_at_25m - defo_at_29m    

        if hydro_year in [2021, 2022]:
            defo_core = defo_at_4_5_m - defo_at_25m
            defo_AL = defo_at_0m - defo_at_4_5_m
            
        else:
            defo_core = defo_at_3_5_m - defo_at_25m
            defo_AL = defo_at_0m - defo_at_3_5_m
        
        defo_shearzone_daily.append(defo_shearzone)
        defo_core_daily.append(defo_core)
        defo_AL_daily.append(defo_AL)
        months.append(date)
        hydro_years.append(hydro_year)
            
    defo_layer_daily = pd.DataFrame({
        'month': months,
        'hydro_year': hydro_years,
        'defo_shearzone': defo_shearzone_daily,
        'defo_core': defo_core_daily,
        'defo_AL': defo_AL_daily
    })

    # Smoothen resolt of daily deformation per layer

    defo_shearzone_daily_smooth = pd.Series(defo_shearzone_daily).rolling(window = 30, center = True).mean()
    defo_core_daily_smooth = pd.Series(defo_core_daily).rolling(window = 30, center = True).mean()
    defo_AL_daily_smooth = pd.Series(defo_AL_daily).rolling(window = 30, center = True).mean()

    defo_layer_daily_smooth = pd.DataFrame({
        'month': months,
        'hydro_year': hydro_years,
        'defo_shearzone': defo_shearzone_daily_smooth,
        'defo_core': defo_core_daily_smooth,
        'defo_AL': defo_AL_daily_smooth
    })
    
    
    # VELOCITY each layer MONTHLY
    
    defo_shearzone_monthly = []
    defo_core_monthly = []
    defo_AL_monthly = []
    months = []
    hydro_years = []

    # Iterate for every month in this year
    for index, row in vel_data['vel_monthly'].iterrows():
        
        date = row.name # get the value of the index (in this case a date)
        year = date.year
        month = date.month
        
        hydro_year = year if month > 8 else year - 1
        
        defo_at_0m = row[80]/10 # cm
        defo_at_3_5_m = row[73]/10
        defo_at_4_5_m = row[71]/10
        defo_at_25m = row[30]/10
        defo_at_29m = row[22]/10 
        
        defo_shearzone = defo_at_25m - defo_at_29m    

        if hydro_year in [2021, 2022]:
            defo_core = defo_at_4_5_m - defo_at_25m
            defo_AL = defo_at_0m - defo_at_4_5_m
            
        else:
            defo_core = defo_at_3_5_m - defo_at_25m
            defo_AL = defo_at_0m - defo_at_3_5_m
        
        defo_shearzone_monthly.append(defo_shearzone)
        defo_core_monthly.append(defo_core)
        defo_AL_monthly.append(defo_AL)
        months.append(date)
        hydro_years.append(hydro_year)
            
    defo_layer_monthly = pd.DataFrame({
        'month': months,
        'hydro_year': hydro_years,
        'defo_shearzone': defo_shearzone_monthly,
        'defo_core': defo_core_monthly,
        'defo_AL': defo_AL_monthly
    })

    # Clip to data period with no NaNs
    defo_layer_monthly_16_23 = defo_layer_monthly[
                                        (defo_layer_monthly['month'] >= pd.to_datetime('2016-05-17')) &
                                        (defo_layer_monthly['month'] <= pd.to_datetime('2023-08-01'))]

    defo_layer_monthly_16_23['month'] = pd.to_datetime(defo_layer_monthly_16_23['month'])
    
    # Ensure data is numeric
    for col in ['defo_shearzone', 'defo_core', 'defo_AL']:
        defo_layer_monthly_16_23[col] = pd.to_numeric(defo_layer_monthly_16_23[col], errors='coerce')



    # DISPLACEMENT each LAYER per CALENDAR year

    defo_shearzone_yearly = []
    defo_core_yearly = []
    defo_AL_yearly = []
    years = []

    # Iterate for every day in this year
    for index, row in vel_data['vel_yearly'].iterrows():

        date = row.name # get the value of the index (in this case a date)
        year = date.year
            
        defo_at_0m = row[80] 
        defo_at_3_5_m = row[73] 
        defo_at_4_5_m = row[71]
        defo_at_25m = row[30]
        defo_at_29m = row[22] # can vary this
        
        defo_shearzone = defo_at_25m - defo_at_29m    

        if year in [2022, 2023]:
            defo_core = defo_at_4_5_m - defo_at_25m
            defo_AL = defo_at_0m - defo_at_4_5_m
            
        else:
            defo_core = defo_at_3_5_m - defo_at_25m
            defo_AL = defo_at_0m - defo_at_3_5_m
        
        defo_shearzone_yearly.append(defo_shearzone)
        defo_core_yearly.append(defo_core)
        defo_AL_yearly.append(defo_AL)
        years.append(year)
            
    defo_layer_yearly = pd.DataFrame({
        'year': years,
        'defo_shearzone': defo_shearzone_yearly,
        'defo_core': defo_core_yearly,
        'defo_AL': defo_AL_yearly
    })

    # Remove first 2016 row since for 2016 there is nothing to subtract to before hand so there is no diff (velo) values

    defo_layer_yearly = defo_layer_yearly.iloc[1:,:] 


    # LAYER-SPECIFIC yearly deformation as a PERCENTAGE

    perc_defo_shearzone_yearly_ext = []
    perc_defo_core_yearly_ext = []
    perc_defo_AL_yearly_ext = []
    years = []

    for index, row in vel_data['vel_yearly'].iterrows():
        
        date = row.name # get the value of the index (in this case a date)
        year = date.year
            
        defo_at_0m = row[80] 
        defo_at_3_5_m = row[73] 
        defo_at_4_5_m = row[71]
        defo_at_25m = row[30]
        
        perc_defo_shearzone = (defo_at_25m / defo_at_0m) * 100


        if hydro_year in [2021, 2022]:
            perc_defo_core = ((defo_at_4_5_m - defo_at_25m) / defo_at_0m) * 100
            perc_defo_AL = ((defo_at_0m - defo_at_4_5_m) / defo_at_0m) * 100
            
        else:
            perc_defo_core = ((defo_at_3_5_m - defo_at_25m) / defo_at_0m) * 100
            perc_defo_AL = ((defo_at_0m - defo_at_3_5_m) / defo_at_0m) * 100
        
        perc_defo_shearzone_yearly_ext.append(perc_defo_shearzone)
        perc_defo_core_yearly_ext.append(perc_defo_core)
        perc_defo_AL_yearly_ext.append(perc_defo_AL)
        years.append(year)
        
    perc_defo_layer_yearly_ext = pd.DataFrame({
        'year': years,
        'perc_defo_shearzone': perc_defo_shearzone_yearly_ext,
        'perc_defo_core': perc_defo_core_yearly_ext,
        'perc_defo_AL': perc_defo_AL_yearly_ext
    })

    # Take the mean across all years

    perc_defo_layer_yearly_ext_num = perc_defo_layer_yearly_ext.drop(['year'], 
                                                             axis = 1)

    perc_defo_layer_yearly_means = perc_defo_layer_yearly_ext_num.mean()
    
    
    return {
        "layer_daily": defo_layer_daily,
        "layer_daily_smooth": defo_layer_daily_smooth,
        "layer_monthly": defo_layer_monthly_16_23,
        "layer_yearly": defo_layer_yearly,
        "perc_layer_yearly": perc_defo_layer_yearly_ext,
        "perc_layer_defo": {
            "AL": round(perc_defo_layer_yearly_means['perc_defo_AL'],1),
            "core": round(perc_defo_layer_yearly_means['perc_defo_core'],1),
            "shear": round(perc_defo_layer_yearly_means['perc_defo_shearzone'],1)
            },
        "layer_mean_vel": {
            "AL": round(SAA_vel_hydro_yearly_mean_AL/10, 1),
            "core": round(SAA_vel_hydro_yearly_mean_core/10,1),
            "shear": round(SAA_vel_hydro_yearly_mean_shear/10,1),
            },
        "strain_rates": {
            "AL": round(SAA_dvdz_AL_annual_mean, 3),
            "core": round(SAA_dvdz_core_annual_mean,3),
            "shear": round(SAA_dvdz_shear_annual_mean,3),
            },
        "strain_rates_times": SAA_dvdz_monthly_times,
        
    }