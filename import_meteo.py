#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:21:10 2025

@author: giuliosaibene
"""

import pandas as pd
import os

### DEFINE FUNCTIONS

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


# ------------------------------------- 

### IMPORT DATA

# Relative file paths to repository (make sure to set working directory to repository Murtel_2015_Data folder!!)

data_dir = "Meteo_data" # subfolder 'SAA_data'

meteo_97_19_path = os.path.join(data_dir, "murtel_level_2_hourly.csv")
meteo_19_23_path = os.path.join(data_dir, "Corvatsch87_met.csv")
perma_xt_meteo_22_path = os.path.join(data_dir, "MetCR6__L2.csv")
perma_xt_meteo_24_path = os.path.join(data_dir, "permaxt_data_dom.csv")
piz_precip_path = os.path.join(data_dir, "corvatsch_precip.txt")
piz_temp_path = os.path.join(data_dir, "corvatsch_temp.txt")


permos_meteo_97_19 = pd.read_csv(meteo_97_19_path)
permos_meteo_19_23 = pd.read_csv(meteo_19_23_path, encoding='utf-8', 
                           parse_dates=['TimeStamp'], na_values=['NAN', 6999])
perma_xt_meteo_22 = pd.read_csv(perma_xt_meteo_22_path)
perma_xt_meteo_24 = pd.read_csv(perma_xt_meteo_24_path, parse_dates = ['TimeStamp'], 
                                na_values = 'NAN')
piz_precip = pd.read_csv(piz_precip_path, sep='\s+', skiprows = 8, encoding='latin1', 
                         low_memory=False)
piz_temp = pd.read_csv(piz_temp_path, sep='\s+', skiprows = 8, encoding='latin1', 
                       low_memory=False)


### CLEANING

# PERMOS

permos_meteo_97_19['date'] = pd.to_datetime(permos_meteo_97_19['date'])

permos_meteo_97_19_useful = permos_meteo_97_19[['date', 'airtemp', 'surftemp', 'snowh', 'longout']]


permos_meteo_19_23.columns = ['date', 'airtemp', 'RH', 'VWND1', 'DWND1', 'VWND1_MAX', 'LWRup', 'LWRdown',
       'LWRnet', 'SWRup', 'SWRdown', 'SWRnet', 'snowh', 'surftemp', 'PLU_SUM10']

permos_meteo_19_23['snowh'] = permos_meteo_19_23['snowh']/100

permos_meteo_19_23_useful = permos_meteo_19_23[['date', 'airtemp', 'surftemp', 'snowh', 'LWRdown']]
permos_meteo_19_23_useful.rename(columns={'LWRdown': 'longout'}, inplace=True)

permos_meteo_97_23 = pd.concat([permos_meteo_97_19_useful, permos_meteo_19_23_useful], ignore_index = True)
permos_meteo_97_23['date'] = pd.to_datetime(permos_meteo_97_23['date'])

permos_meteo_97_23 = permos_meteo_97_23[permos_meteo_97_23['date'] <= pd.to_datetime('2023-09-20')]

daily_means = aggregate_to_daily(permos_meteo_97_23)

# Add back missing rows from 2019-04-01 to 2019-09-30
missing_dates = pd.date_range(start='2019-04-01', end='2019-09-30', freq='D')
daily_means.index = pd.to_datetime(daily_means.index)
daily_means_complete = daily_means.reindex(daily_means.index.union(missing_dates))


# PERMA-XT

perma_xt_meteo_24.rename(columns = {'TimeStamp' : 'date'}, inplace = True)

perma_xt_meteo_22.columns = ['date', 'rec_num', 'amb_press', 'airtemp', 'RH', 'airtemp_107probe', 'pluvio', 'snow_height' , 'snow_surf_temp', 
                          'airtemp_25cm_above', 'airtemp_50cm_above', 'airtemp_100cm_above', 'gst_70cm', 'RH_70cm', 'gt_2m', 'RH_2m']

daily_means_perma_xt_22 = aggregate_to_daily(perma_xt_meteo_22)
daily_means_perma_xt_24 = aggregate_to_daily(perma_xt_meteo_24)

perma_xt_meteo_24.set_index('date', drop = False, inplace = True)
daily_sums_perma_xt_24 = perma_xt_meteo_24['Pluvio'].resample('D').sum()
# Add column back to daily_means_perma_xt_24 
daily_means_perma_xt_24['Pluvio_sum'] = daily_sums_perma_xt_24


# MeteoSwiss Piz Corvatsch

piz_precip.columns = ['STA', 'JAHR', 'MO', 'TG', 'HH', 'MM', 'precip']

piz_precip['date'] = pd.to_datetime(piz_precip[['JAHR', 'MO', 'TG']].astype(str).agg('-'.join, axis=1))
piz_temp['date'] = pd.to_datetime(piz_temp[['JAHR', 'MO', 'TG']].astype(str).agg('-'.join, axis=1))

piz_precip = piz_precip.drop(['STA', 'HH', 'MM'], axis = 1)
piz_temp = piz_temp.drop(['STA', 'HH', 'MM'], axis = 1)

piz_precip['date'] = pd.to_datetime(piz_precip['date'], errors='coerce')
piz_temp.rename(columns = {'211':'temp'}, inplace = True)

piz_precip = piz_precip.drop(index=0)
piz_temp = piz_temp.drop(index=0)

# Remove values unrealistically high
piz_precip = piz_precip[piz_precip['precip'] <= 500]

### Export to CSV

daily_means_complete_path = os.path.join(data_dir, "daily_means_permos.csv")
daily_means_complete.to_csv(daily_means_complete_path, index=False)

daily_means_perma_xt_24_path = os.path.join(data_dir, "daily_means_perma_xt.csv")
daily_means_perma_xt_24.to_csv(daily_means_perma_xt_24_path, index=False)

