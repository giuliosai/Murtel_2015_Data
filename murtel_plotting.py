#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:02:12 2025

@author: giuliosaibene
"""

# =============================================================================
# USE THE PROCESSED DATA TO GENERATE FIGURES FROM THE SAIBENE ET AL. (2025) MANUSCRIPT
# =============================================================================

from process_SAA_data import load_and_prepare_saa_data, compute_velocities, compute_layer_deformation
from process_other_data import load_and_prepare_gnss_data, load_and_prepare_geodetic_data, load_and_prepare_temp_data, load_and_prepare_meteo_data, prepare_corr_data

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os
from matplotlib import gridspec


## IMPORT data using functions from seperate script

SAA_data = load_and_prepare_saa_data()
SAA_vel = compute_velocities(SAA_data)
SAA_layers = compute_layer_deformation(SAA_data, SAA_vel)

GNSS_data = load_and_prepare_gnss_data()
geodetic_data = load_and_prepare_geodetic_data()
temp_data = load_and_prepare_temp_data()
meteo_data = load_and_prepare_meteo_data(temp_data)

corr_data = prepare_corr_data(GNSS_data, temp_data, meteo_data)

## PLOTTING FUNCTIONS

# Plot deformation profile (multiple years and each month lined color by month)

# Can plot either deformation (defoSwitch = True) or velocity (velSwitch) or strain rate (strainSwitch)

def plot_vertical_profile_single_months(total_defo, z, times, nlines, start_date, 
                                  end_date, min_x, max_x, 
                                  defoSwitch, velSwitch, strainSwitch,
                                  save):
    
    # defoSwitch is needed for plotting deformation profiles
    # velSwitch is needed for plotting velocity profiles
    # strainSwitch is needed for plotting strain profiles
    
    fs14 = 14
    
    times = times.dt.date
    
    # Making sure index starts from 0
    if times.index[0] != 0:
        times.reset_index(inplace = True, drop = True)

    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    # First check is start and end dates are actually in times input
    if start_date not in times.values:
        print(f"Warning: Start date {start_date} not found in times.")
    else:
        tini = times[times == start_date].index[0]  # Gets the start index
    
    if end_date not in times.values:
        print(f"Warning: End date {end_date} not found in times.")
    else:
        tend = times[times == end_date].index[0]  # Gets the end index

    # Making sure index starts from 0
    if total_defo.index[0] != 0:
        total_defo.reset_index(inplace = True, drop = True)

    # Remove level_0 column from previous resetting, before resetting index
    if 'level_0' in total_defo.columns:
        total_defo.drop(columns='level_0', inplace=True)
    
    if 'time' not in total_defo.columns:
        # Reset index to make sure it has numeric values
        total_defo.reset_index(drop = False, inplace = True)
    
    # Only select given time frame specified by start and end dates
    total_defo_period = total_defo[ (total_defo.index >= tini) & (total_defo.index <= tend) ]
    
    if defoSwitch: # To subtract initial value
        
        # Make sure only numeric columns are taken
        non_numeric_cols = total_defo_period.select_dtypes(exclude=['number']).columns
        
        total_defo_period_num = total_defo_period.drop(columns = non_numeric_cols)
    
        # Subtract initial value from all values across all depths (columns)
        total_defo_period_rel = total_defo_period_num.subtract(total_defo_period_num.iloc[0],
                                                           axis = 1)
        
        # Remove non-depth columns (ones that have letters in their column names)
    
        total_defo_period_rel.columns = total_defo_period_rel.columns.astype(str)
    
        total_defo_period_rel_num = total_defo_period_rel.loc[:, ~total_defo_period_rel.columns.str.contains(r'[a-zA-Z]')]
        
    else: # For plotting velocity or strain rate profiles no need to subtract initial value
    
        # Remove non-depth columns (ones that have letters in their column names)
    
        total_defo_period.columns = total_defo_period.columns.astype(str)
    
        total_defo_period_rel_num = total_defo_period.loc[:, ~total_defo_period.columns.str.contains(r'[a-zA-Z]')]
        
    # Set the index of z to numeric
    
    z_num = z.reset_index(drop = True, inplace = False)
        
    timestep = np.round(np.linspace(tini, tend, nlines)).astype(int)  # Vector of timesteps
        
    dates = times.iloc[timestep]
    
    # Create monthly color map
    
    month_colors = ['#DC6E42', '#BD3B3B', '#BC369C', '#9748C1', '#46388E',
                '#5567DB', '#4D88C2', '#4AB3CD', '#3FA597', '#B3AF56',
                '#FAD274', '#FFB166']
    months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    
    # Starting month index
    start_month_idx = months.index('May') # input times starts in 2016-06 (so put the month before due to 0-based indexing)

    # Calculate the month for each timestep
    month_indices = [(start_month_idx + i) % len(months) for i in range(nlines)]

    color_map = [month_colors[idx] for idx in month_indices]
    
    # Plot
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(nlines):
        
        line, = ax.plot(total_defo_period_rel_num.loc[timestep[i], :], abs(z_num.iloc[timestep[i], :]), 
                         color = color_map[i])
        
    ax.set_xlim([min_x, max_x])
        
    if velSwitch:
        ax.set_xlabel('Velocity (mm/month)', fontsize=fs14)
        
    elif strainSwitch:
        ax.set_xlabel(r'Strain rate ($month^{-1}$)', fontsize=fs14)
        
    else:
        ax.set_xlabel('Displacement (cm)', fontsize=fs14)
        
    ax.set_ylabel('Depth (m)', fontsize = 16)
        
    ax.invert_yaxis()    
    
    ax.tick_params(labelsize=fs14, width=1)
    
    legend_handles = [mpatches.Patch(color=month_colors[i], label=months[i]) for i in range(len(months))]
    
    ax.legend(handles=legend_handles, loc="lower right", fontsize = 14)
    
    ax.grid(True, color = 'grey', linestyle = '--', alpha = 0.7)
    
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', color='gray', linewidth=0.8, alpha=0.6) 
    ax.tick_params(which='minor', length=0)

    ax.set_ylim(42,-2)
    
    if save:
        
        # Extract the years
        
        start_yr = start_date.year
        end_yr = end_date.year
        
        defo_fig_path = os.path.join("SAA_data", f"defo_profile_months_{start_yr}_{end_yr}.png")
    
        plt.savefig(defo_fig_path, dpi = 300, bbox_inches = 'tight')
        



# Plotting the layer-specific deformation

def plot_defo_layers(
    gnss_diff_yearly,
    geodetic_diff_yearly,
    defo_layer_yearly,
    gnss_diff_monthly,
    defo_layer_monthly,
    color_shearzone="#713601",
    color_core="#F27202",
    color_AL="#FEBE86",
    bar_width=1,
    figsize=(10, 6),
    save_path=None
):
    """
    Create two subplots of annual and monthly deformation rates comparing GNSS, geodetic, and layer contributions.
    
    Parameters
    ----------
    gnss_diff_yearly : pd.DataFrame of GNSS annual velocity data
        Must contain ['year', 'horiz_displacement_filt'].
    geodetic_diff_yearly : pd.DataFrame of geodetic (total station) annual velocity data
        Must contain ['decimal_year', 'defo'].
    defo_layer_yearly : pd.DataFrame of annual velocity within each layer from inclinometer
        Must contain ['year', 'defo_shearzone', 'defo_core', 'defo_AL'].
    gnss_diff_monthly : pd.DataFrame of GNSS monthly velocity data
        Must contain ['time', 'horiz_displacement_filt'].
    defo_layer_monthly_smooth : pd.DataFrame of layer-specific monthly velocity data from inclinometer
        Must contain ['month', 'defo_shearzone', 'defo_core', 'defo_AL'].
    color_shearzone, color_core, color_AL : str
        Colors for the stacked plots.
    bar_width : float
        Width of the yearly stacked bars.
    figsize : tuple
        Figure size for the whole plot.
    save_path : str or None
        If given, saves the figure to this path.
    """

    fig, axs = plt.subplots(2, 1, figsize=figsize)

    # ---------------- Annual ----------------
    axs[0].step(
        gnss_diff_yearly['year'],
        gnss_diff_yearly['horiz_displacement_filt'] * 100,  # mm
        where='post',
        linewidth=2,
        color='black',
        label='GNSS'
    )

    axs[0].step(
        geodetic_diff_yearly['decimal_year'],
        geodetic_diff_yearly['defo'] * 100,  # mm
        where='post',
        linewidth=2,
        linestyle='--',
        color='k',
        label="Geodetic",
        alpha=1
    )

    axs[0].bar(
        defo_layer_yearly['year'],
        defo_layer_yearly['defo_shearzone'] / 10,
        label='Shear zone',
        color=color_shearzone,
        width=bar_width,
        align='edge'
    )

    axs[0].bar(
        defo_layer_yearly['year'],
        defo_layer_yearly['defo_core'] / 10,
        bottom=defo_layer_yearly['defo_shearzone'] / 10,
        label='Core',
        color=color_core,
        width=bar_width,
        align='edge'
    )

    axs[0].bar(
        defo_layer_yearly['year'],
        defo_layer_yearly['defo_AL'] / 10,
        bottom=defo_layer_yearly['defo_shearzone'].reset_index(drop=True) / 10 +
               defo_layer_yearly['defo_core'].reset_index(drop=True) / 10,
        label='AL',
        color=color_AL,
        width=bar_width,
        align='edge'
    )

    axs[0].set_ylabel('Annual velocity\n(cm/year)', size=14)
    axs[0].legend(ncol=5, fontsize=13, loc='lower left')
    axs[0].grid(True, alpha=0.6, linestyle='--')
    axs[0].set_axisbelow(True)
    axs[0].set_xticklabels([])
    axs[0].set_xlim(2017, 2023)
    axs[0].text(
        0.015, 0.89, '(a)', size=12, transform=axs[0].transAxes,
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
    )

    # ---------------- monthly ----------------
    axs[1].plot(
        pd.to_datetime(gnss_diff_monthly['time']),
        gnss_diff_monthly['horiz_displacement_filt'] * 100,  # cm
        color='black',
        linewidth=1,
        label='GNSS'
    )

    # Shear zone
    axs[1].plot(
        defo_layer_monthly['month'],
        defo_layer_monthly['defo_shearzone'],
        label="Shear zone",
        color=color_shearzone
    )
    axs[1].fill_between(
        defo_layer_monthly['month'],
        defo_layer_monthly['defo_shearzone'],
        color=color_shearzone, alpha=0.7
    )

    # Ice core
    axs[1].plot(
        defo_layer_monthly['month'],
        defo_layer_monthly['defo_shearzone'] + defo_layer_monthly['defo_core'],
        label="Ice core",
        color=color_core
    )
    axs[1].fill_between(
        defo_layer_monthly['month'],
        defo_layer_monthly['defo_shearzone'],
        defo_layer_monthly['defo_shearzone'] + defo_layer_monthly['defo_core'],
        color=color_core, alpha=0.7
    )

    # AL
    axs[1].plot(
        defo_layer_monthly['month'],
        defo_layer_monthly['defo_shearzone'] +
        defo_layer_monthly['defo_core'] +
        defo_layer_monthly['defo_AL'],
        label="AL",
        color=color_AL
    )
    axs[1].fill_between(
        defo_layer_monthly['month'],
        defo_layer_monthly['defo_shearzone'] + defo_layer_monthly['defo_core'],
        defo_layer_monthly['defo_shearzone'] +
        defo_layer_monthly['defo_core'] +
        defo_layer_monthly['defo_AL'],
        color=color_AL, alpha=0.7
    )

    axs[1].axhline(0, color='grey', linewidth=0.6, alpha=0.6)
    axs[1].set_ylabel("Monthly velocity\n(cm/month)", size=14)
    axs[1].set_xlabel("Year", size=14)
    axs[1].set_xlim(pd.to_datetime("2017-01-01"), pd.to_datetime("2023-01-01"))
    axs[1].tick_params(axis='x', labelsize=12)
    axs[1].grid(True, alpha=0.6, linestyle='--')
    axs[1].set_axisbelow(True)
    axs[1].text(
        0.015, 0.89, '(b)', size=12, transform=axs[1].transAxes,
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
    )

    plt.subplots_adjust(hspace=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axs



## Annual deformation vertical profiles
    
def plot_vertical_profile_annual_cal(total_defo, z, years, min_x, max_x, save,
                                     ax = None):
    
    # Clean data frame
    
    total_defo_clean = total_defo.copy()
    for col in ['julian_day_hydro', 'date']:
        if col in total_defo_clean.columns:
            total_defo_clean = total_defo_clean.drop(columns=col)
        
    # Reset the index of z to numeric
    
    z_num = z.reset_index(drop = True, inplace = False)
        
    #print("total_defo_period_rel_num index:", total_defo_period_rel_num.index)
    #print("z index:", z_num.index)
        
    timestep = np.round(np.linspace(2016, 2022, 7)).astype(int)  # Vector of years
    
    #print("timestep:", timestep)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    year_colors_17_22 = ['#46388E','#4D88C2', '#3FA597', 
                   '#FAD274', '#DC6E42', '#d10845']
    
    for i, year in enumerate(years):
                
        line, = ax.plot(total_defo_clean.loc[year, :].to_numpy().flatten()/10, abs(z_num.iloc[1, :]), 
                         color = year_colors_17_22[i], label = years[i])
        
    ax.invert_yaxis()
    
    ax.set_ylim(abs(z_num.iloc[-1, 1]), 0) # Starting y axis from lowest depth
    
    #ax.set_ylim(40, 0)
    
    # Ticks and labels
    
    #ticks = ax.get_xticks()[1:-1] # Remove last tick
    ticks = ax.get_xticks()
    
    #int_ticks = [int(round(tick)) for tick in ticks]
    formatted_ticks = [f"{tick:.1f}".rstrip('0').rstrip('.') for tick in ticks]
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(formatted_ticks)
    
    ax.xaxis.set_ticks_position('both')  # Show ticks on both top and bottom
    ax.tick_params(bottom = True, labelbottom = False, top=True, labeltop=True)
    
    ax.set_xlabel('Annual deformation velocity (cm/year)', fontsize = 14)
    
    # Titles and legend
        
    ax.legend(loc='upper left', fontsize=14,
              ncol = 1, columnspacing = 1.5)
        
    ax.set_ylabel('Depth (m)', fontsize=22)
            
    ax.tick_params(labelsize = 18, width=1)
    ax.tick_params(axis="y", labelleft=True)
    
    ax.minorticks_on()
    
    ax.grid(True, which = 'major')
    ax.grid(True, which='minor', linestyle=':', color='gray', linewidth=0.5, alpha=0.5)
    
    # X Axis
    
    ax.set_xlim([min_x, max_x])
    
    if save:
    
        plt.savefig('/Users/giuliosaibene/Desktop/University/UZH/Thesis/Figures/profiles_annual_defo_vert_cal.png', 
                    dpi = 300, bbox_inches = 'tight')
        
        
## Annual mean temperature vertical profiles

def plot_annual_temp_profiles_17(temp_annual_means, depth_str, 
                              cbarSwitch, ax=None):
    # Clean temp df if needed
    temp_annual_means_clean = temp_annual_means.copy()

    for col in ['time','date', 'julian_day_hydro', 'phase', 'sem_julian_day_hydro']:
        if col in temp_annual_means_clean.columns:
            temp_annual_means_clean = temp_annual_means_clean.drop(col, axis=1)

    # Convert depth string to list of floats
    depth = [float(x.strip("'")) for x in depth_str.split(", ")]

    # Extract the range of years
    start_year = min(temp_annual_means_clean.index.year)
    end_year = max(temp_annual_means_clean.index.year)

    # Define custom colors for each year
    year_colors = ['#46388E', '#4D88C2', '#3FA597', 
                   '#FAD274', '#DC6E42', '#d10845']
    
    years = list(range(start_year, end_year + 1))
    
    if len(years) > len(year_colors):
        raise ValueError("Not enough colors in year_colors for the number of years.")

    # Create a figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))

    # Plot each year's data with the corresponding color
    for i, year in enumerate(years):
        if str(year) in temp_annual_means_clean.index:
                        
            ax.plot(temp_annual_means_clean.loc[str(year)], depth, 
                    marker='o', color=year_colors[i], markersize=2, label=str(year))

    # Draw 0˚C vertical line
    #ax.axvline(x=0, color='black', linewidth=2, linestyle='--')

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()
    
    ax.set_ylim(40,0)

    if ax is None:
        # Add color legend instead of colorbar
        ax.legend(title="Year", fontsize=10)
        ax.set_ylabel('Depth (m)', fontsize=14)

    # Add labels
    ax.set_xlabel('Temperature (°C)', fontsize=14)
    
    ax.minorticks_on()
    
    ax.tick_params(labelsize = 18, width=1)
    
    ax.grid(True, which = 'major')
    ax.grid(True, which='minor', linestyle=':', color='gray', linewidth=0.5, alpha=0.5)
    
    if ax is None:
        ax.tick_params(axis = 'y', length = 0, labelsize = 18)
        ax.set_yticklabels([])
        
    ax.xaxis.set_ticks_position('both')  # Show ticks on both top and bottom
    ax.tick_params(bottom = True, labelbottom = False, top=True, labeltop=True)

    return ax

def plot_annual_defo_temp_profiles(
        vel_yearly, z_daily, 
        temp_profiles, temp_depths,
        years=None, min_x=-2, max_x=18,
        figsize=(10, 6), save=False, outpath=None):
    """
    Combine annual deformation and temperature profiles into a single figure.
    
    Parameters
    ----------
    vel_yearly : pd.DataFrame or array
        Annual deformation velocities.
    z_daily : array
        Depths for the deformation data.
    temp_profiles : pd.DataFrame or array
        Annual temperature profiles.
    temp_depths : array
        Depths for temperature data.
    years : list of str, optional
        Years to plot. Default None = use all.
    min_x, max_x : float
        X-axis limits for deformation plot.
    figsize : tuple
        Figure size.
    save : bool
        Whether to save the figure.
    outpath : str
        Path to save the figure if save=True.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # Deformation plot
    plot_vertical_profile_annual_cal(
        vel_yearly, z_daily,
        years=years, min_x=min_x, max_x=max_x,
        save=False, ax=axs[0]
    )
    
    # Temperature plot
    plot_annual_temp_profiles_17(
        temp_profiles, temp_depths,
        cbarSwitch=True, ax=axs[1]
    )
    
    # Labels (a) and (b)
    axs[0].text(0.91, 0.03, '(a)', size=12, transform=axs[0].transAxes,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    axs[1].text(0.91, 0.03, '(b)', size=12, transform=axs[1].transAxes,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    
    plt.subplots_adjust(wspace = 0)
    
    if save and outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
    
    return fig, axs

def plot_ro3_daily_ZCsbars_Qg_Qm_vel(temp_hole_15, temp_hole_15_layer, daily_means, 
                                       early_winter, precip, spring_ZC_times, ground_heat_flux, 
                                       melt_nrg, defo_layer, surf_vel, snow_end_time,
                                       freezing_onset_time, Save, outpath=None):
    
    """
    Plot the environmental time series (air temperature, snow height, 
                                        precipitation, borehole temps)
    in combination with the ground heat flux and snowmelt energy flux and 
    lastly the velocity time series including the layer-specific borehole 
    data and the GNSS monthly data.
    
    Shade in the warm, cold phases and spring zero curtains across all subplots.
    
    The duration of the spring ZC is added as bars in the bottom subplot.
    """

    nplots = 3
    
    fig, axs = plt.subplots(nplots, 1, figsize=(10, 8), 
                            gridspec_kw={'height_ratios': [3, 1, 2]})
    
    span_handles = {} # for custom legend items
    
    # Precipitation 
    ax2 = [None] * 2
    ax2[0] = axs[0].twinx()
    
    # Convert snow height (m) to SWE (mm w.e.)
    ax2[0].bar(early_winter['date'], early_winter['snowh']*0.3*100, width = 7, 
               label = 'Early winter snow', color = '#27a0cc', zorder = 1)
    
    ax2[0].bar(daily_means['date'], daily_means['snowh']*0.3*100, width = 7, 
               label = 'Snow', color = "#73d2de", zorder = 0)
    
    # Precipitation from warm phase
    ax2[0].bar(precip['date'], precip['precip'], width = 3, color = "#0725b8",
               label = "Rainfall", alpha = 1)
    
    ax2[0].set_ylabel("Precipitation (mm)", size = 14, color = '#213e96')
    
    ax2[0].tick_params(axis='y', colors='#213e96')
    
    ax2[0].invert_yaxis()
    #ax2[0].set_ylim(4, 0)
    ax2[0].set_ylim(140, 0)
    
    # Remove last tick
    ax2[0].set_yticklabels([int(tick) for tick in ax2[0].get_yticks()[:-3]])
    
    span_handles['Early winter snow'] = Patch(color='#27a0cc', label='Early winter snow', linewidth=0, 
                                      edgecolor='none')
    span_handles['Snow'] = Patch(color="#73d2de", label='Snow', linewidth=0, 
                                      edgecolor='none')
    span_handles['Rainfall'] = Patch(color="#0725b8", label='Rainfall', linewidth=0, 
                                      edgecolor='none')
    
    # Temperature
    
    axs[0].plot(daily_means['date'], daily_means['airtemp_smooth'], label = "Air",
                color = "#3096B0", linewidth = 0.8, alpha = 0.6)
    
    axs[0].plot(temp_hole_15['time'], temp_hole_15['0.25'].rolling(30).mean(), 
                label = '0.25 m', color = '#bf1553') #a69800

    axs[0].plot(temp_hole_15['time'], temp_hole_15['3.01'], label = '3 m',
                color = "#F27202")
    
    axs[0].plot(temp_hole_15['time'], temp_hole_15['8'], label = '8 m',
                color = '#d6962f') #a82f14
    
    # Shade warm phase periods based on snow end date and freezing onset time
    
    snow_end_time_16_23 = snow_end_time[snow_end_time['HydroYear'] >= 2015]
    
    for i_plot in np.arange(nplots):
    
        for i in range(len(snow_end_time_16_23)):
            axs[i_plot].axvspan(snow_end_time_16_23['LastSnowDate'].iloc[i], 
                       freezing_onset_time['first_freezing_day'].iloc[i+1], 
                       color='red', alpha=0.1)
                    
        # Shade cold phase periods based on snow end date and freezing onset time
    
        for i in range(len(snow_end_time_16_23) - 1):
            axs[i_plot].axvspan(freezing_onset_time[:-1]['first_freezing_day'].iloc[i+1], # take only up to 2022-11 for last freezing date
                       snow_end_time_16_23[1:]['LastSnowDate'].iloc[i], # take only starting from 2017-06 last snow date
                       color='blue', alpha=0.1)
            
    axs[0].set_ylim(-13, 20)
    #axs[0].set_ylim(-7.5, 15)
    axs[0].set_ylabel("Mean Temperature (˚C)", size = 15)
    
    # Removing last tick label
    
    yticks = axs[0].get_yticks()
    axs[0].set_yticks(yticks[1:])  
    
    # Add T = 0˚C line        
    axs[0].axhline(y = 0, linewidth = 1, alpha = 0.3, color = 'grey')

    
    # ZC and phase durations
    
    for i_plot in np.arange(nplots):
        for i in range(len(spring_ZC_times)):
            axs[i_plot].axvspan(spring_ZC_times['start_date'].iloc[i], spring_ZC_times['end_date'].iloc[i],
                           color="#db6618", alpha = 0.2)
    
    # May 1 tick
    
    for year in np.arange(2017, 2024, 1):
        axs[0].plot([pd.to_datetime(f'{year}-05-01'), pd.to_datetime(f'{year}-05-01') ], 
                    [-13, -12.3], color = 'k', linewidth = 1)
        
    for year in np.arange(2017, 2024, 1):
        axs[1].plot([pd.to_datetime(f'{year}-05-01'), pd.to_datetime(f'{year}-05-01') ], 
                    [-5, -3.5], color = 'k', linewidth = 1)
        
    for year in np.arange(2017, 2024, 1):
        axs[2].plot([pd.to_datetime(f'{year}-05-01'), pd.to_datetime(f'{year}-05-01') ], 
                    [-0.1, -0.03], color = 'k', linewidth = 1)
        
    axs[0].text(0.408, 0.05, 'May-1', size = 10, color = 'grey',
            transform = axs[0].transAxes, rotation = 'vertical')
    
    axs[0].text(0.26, 0.5, 'Cold\n phase', size = 11, color = "#0f5fba",
                ha = 'center', transform = axs[0].transAxes)
    
    axs[0].text(0.33, 0.03, 'Warm phase', size = 11, color = "#db6618",
                rotation = 'vertical', ha = 'center', transform = axs[0].transAxes)
    
    span_handles['Spring ZC'] = Patch(color="#db6618", alpha=0.2, 
                                      label='Spring ZC', linewidth=0, 
                                      edgecolor='none')
        
    # Legend
    
    handles, labels = axs[0].get_legend_handles_labels() # get normal labels
    handles.extend(span_handles.values())   # Add the span handles
    axs[0].legend(handles=handles, ncol = 5, bbox_to_anchor = (0.5, 1.13), 
                  loc = 'center', fontsize = 14, columnspacing = 1.5)
    
    # Ground heat flux
    
    axs[1].axhline(y = 0, linewidth = 1, alpha = 0.3, color = 'grey')

    axs[1].plot(temp_hole_15['time'], ground_heat_flux['3.5'], label = '3.5 m',
                color = 'black')
            
    axs[1].set_ylabel("$Q_{G}$\n(Wm$^{-2}$)", size = 14)
    
    axs[1].set_ylim(-5, 15)
    
    axs1_2 = axs[1].twinx()
    
    axs1_2.plot(melt_nrg['date'], melt_nrg['melt_nrg'], linewidth = 1,
                color = "#24b5c7")
    axs1_2.set_ylabel("$Q_{M}$\n(Wm$^{-2}$)", size = 14, color = '#24b5c7')
    
    axs1_2.tick_params(axis='y', colors='#24b5c7')
    
    axs1_2.invert_yaxis()
    
    axs1_2.set_ylim(170, 0)

    # Velocity
    
    axs[2].plot(pd.to_datetime(surf_vel['time']), surf_vel['horiz_displacement_filt']*100, 
                color = 'black', linestyle = '--', linewidth = 0.6, label = 'GNSS')

    axs[2].plot(defo_layer['month'], defo_layer['defo_shearzone'],
            label = "Shear zone", color = "#713601")

    axs[2].fill_between(defo_layer['month'], 
                    defo_layer['defo_shearzone'], 
                    color="#713601", alpha=0.7)

    axs[2].plot(defo_layer['month'], 
            defo_layer['defo_shearzone'] + defo_layer['defo_core'],
            label = "Ice core", color = "#F27202")

    axs[2].fill_between(defo_layer['month'], 
                    defo_layer['defo_shearzone'], 
                    defo_layer['defo_shearzone'] + defo_layer['defo_core'], 
                    color="#F27202", alpha=0.7)

    axs[2].plot(defo_layer['month'], 
            defo_layer['defo_shearzone'] + defo_layer['defo_core'] + defo_layer['defo_AL'],
            label = "AL", color = "#FEBE86")

    axs[2].fill_between(defo_layer['month'], 
                    defo_layer['defo_shearzone'] + defo_layer['defo_core'], 
                    defo_layer['defo_shearzone'] + defo_layer['defo_core'] + defo_layer['defo_AL'], 
                    color="#FEBE86", alpha=1)

    axs[2].axhline(0, color = 'grey', linewidth = 0.6, alpha = 0.6)

    axs[2].set_ylabel("Velocity\n(cm/month)", size = 14)

    axs[2].legend(ncol = 4, columnspacing = 1, fontsize = 11)
    
    axs[2].tick_params(axis = 'x', labelsize = 13)
    
    # ZC durations
    
    axs2_2 = axs[2].twinx()
    
    # Calculate the width for each bar
    spring_ZC_times['width'] = spring_ZC_times['end_date'] - spring_ZC_times['start_date']
    
    axs2_2.bar(
        spring_ZC_times['start_date'], 
        spring_ZC_times['ZC_duration'], 
        width = spring_ZC_times['width'], 
        color = '#787878', 
        align='edge', alpha = 0.5
    )

    axs2_2.set_ylabel("Spring ZC \n duration (days)", size = 14,
                      color = '#787878')
    
    axs2_2.tick_params(axis='y', colors='#787878')
    
    axs2_2.set_ylim(0, 50)
    
    
    # Set x limits
    for ax in axs:
        
        ax.set_xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2023-12-31'))
        #ax.set_xlim(pd.to_datetime('2021-01-01'), pd.to_datetime('2023-06-16'))
    
    # Set minor grid and remove minor x-ticks
    for ax in axs:
        ax.grid(True, linestyle = '--', color = 'gray', alpha = 0.6)
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='x', linestyle=':', color='gray', linewidth=0.5, alpha=0.8) 
        ax.tick_params(which='minor', length=0)  # Hide minor ticks
    
    # Remove x-axis ticks and labels for all but the last subplot
    for ax in axs[:-1]:
        ax.tick_params(axis='x', which='both', length=0)   
        ax.set_xticklabels([])  
    
    plt.subplots_adjust(hspace=0)
    
    if Save and outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')

def plot_spring_melt_deformation_corrs(corr_data, figsize=(14, 6), Save = False,
                                       outpath = None):
    """
    Produce the 6-panel figure showing spring melt dynamics and GNSS/ALT correlations.
    
    Parameters
    ----------
    corr_data : dict or dataframe-like
        Dictionary containing all correlation inputs and regression results.
    figsize : tuple
        Figure size in inches.
    """

    plt.figure(figsize=figsize)

    # Grid layout
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.4, hspace=0.4)

    # Axes definitions
    ax1 = plt.subplot(gs[0, :2])
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[1, 2])
    ax6 = plt.subplot(gs[1, 3])

    # --------------------------
    # (a) 1987: End SZC vs AL temp
    # --------------------------
    ax1.scatter(corr_data['spring_ZCs_87_end_dates'],
                corr_data['annual_max_AL_temp_87'], color='k')

    ax1.plot(corr_data['spring_ZCs_87_end_dates'],
             corr_data['end_SZC_AL_T87_reg']['slope'] * corr_data['spring_ZCs_87_end_dates']
             + corr_data['end_SZC_AL_T87_reg']['intercept'], color='k')

    ax1.text(0.05, 0.35, '1987 borehole', size=12, transform=ax1.transAxes)
    ax1.text(0.05, 0.20, f"R$^2$ = {corr_data['r2_end_SZC_AL_T87']}", size=12,
             transform=ax1.transAxes)
    ax1.text(0.05, 0.05,
             f"p-value = {corr_data['end_SZC_AL_T87_reg']['p_value']:.2f}",
             size=12, transform=ax1.transAxes)
    ax1.text(0.94, 0.9, '(a)', size=12, transform=ax1.transAxes)
    ax1.set_xlabel("End of spring ZC (DOY)", size=14)
    ax1.set_ylabel("Annual max mean \nAL temperature (˚C)", size=14)

    # --------------------------
    # (b) 2015: End SZC vs ALT max date
    # --------------------------
    ax2.scatter(corr_data['spring_ZCs_15_end_dates'],
                corr_data['max_ALT_date'], color='k')

    ax2.plot(corr_data['spring_ZCs_15_end_dates'],
             corr_data['end_SZC_ALT_max_date_reg']['slope'] 
             * corr_data['spring_ZCs_15_end_dates']
             + corr_data['end_SZC_ALT_max_date_reg']['intercept'], color='k')

    ax2.text(0.05, 0.35, '2015 borehole', size=12, transform=ax2.transAxes)
    ax2.text(0.05, 0.20, f"R$^2$ = {corr_data['r2_end_SZC_ALT_max_date']}", size=12,
             transform=ax2.transAxes)
    ax2.text(0.05, 0.05,
             f"p-value = {corr_data['end_SZC_ALT_max_date_reg']['p_value']:.2f}",
             size=12, transform=ax2.transAxes)
    ax2.text(0.94, 0.9, '(b)', size=12, transform=ax2.transAxes)
    ax2.set_xlabel("End of spring ZC (DOY)", size=14)
    ax2.set_ylabel("Date of max ALT (DOY)", size=14)

    # --------------------------
    # (c) End SZC vs GNSS velocity
    # --------------------------
    ax3.scatter(corr_data['spring_ZCs_15_end_dates'],
                corr_data['annual_max_gnss_displacement'] * 1000, color='k')

    ax3.plot(corr_data['spring_ZCs_15_end_dates'],
             corr_data['end_SZC_GNSS_reg']['slope'] 
             * corr_data['spring_ZCs_15_end_dates']
             + corr_data['end_SZC_GNSS_reg']['intercept'], color='k')

    ax3.text(0.05, 0.35, '2015 borehole', size=12, transform=ax3.transAxes)
    ax3.text(0.05, 0.20, f"R$^2$ = {corr_data['r2_end_SZC_GNSS']}", size=12,
             transform=ax3.transAxes)
    ax3.text(0.05, 0.05,
             f"p-value = {corr_data['end_SZC_GNSS_reg']['p_value']:.2f}",
             size=12, transform=ax3.transAxes)
    ax3.text(0.01, 1.05, '(c)', size=12, transform=ax3.transAxes)
    ax3.set_xlabel("End of spring ZC (DOY)", size=14)
    ax3.set_ylabel("Annual max GNSS \nvelocity (mm/d)", size=14)

    # --------------------------
    # (d) Date of max Qm vs GNSS vel
    # --------------------------
    ax4.scatter(corr_data['snowmelt_max_time_16_22'],
                corr_data['annual_max_gnss_displacement'] * 1000, color='k')

    ax4.plot(corr_data['snowmelt_max_time_16_22'],
             corr_data['Qm_maxtime_GNSS_reg']['slope'] 
             * corr_data['snowmelt_max_time_16_22']
             + corr_data['Qm_maxtime_GNSS_reg']['intercept'], color='k')

    ax4.text(0.05, 0.20, f"R$^2$ = {corr_data['r2_Qm_maxtime_GNSS']}",
             size=12, transform=ax4.transAxes)
    ax4.text(0.05, 0.10,
             f"p-value = {corr_data['Qm_maxtime_GNSS_reg']['p_value']:.0e}",
             size=12, transform=ax4.transAxes)
    ax4.text(0.01, 1.05, '(d)', size=12, transform=ax4.transAxes)
    ax4.set_xlabel("Date of max snowmelt (DOY)", size=12)

    # --------------------------
    # (e) AL temp vs GNSS vel
    # --------------------------
    ax5.scatter(corr_data['annual_max_AL_temp_15'],
                corr_data['annual_max_gnss_displacement'] * 1000, color='k')

    ax5.plot(corr_data['annual_max_AL_temp_15'],
             corr_data['AL_temp_GNSS_reg']['slope']
             * corr_data['annual_max_AL_temp_15']
             + corr_data['AL_temp_GNSS_reg']['intercept'], color='k')

    ax5.text(0.05, 0.90, f"R$^2$ = {corr_data['r2_AL_temp_GNSS']}",
             size=12, transform=ax5.transAxes)
    ax5.text(0.05, 0.80,
             f"p-value = {corr_data['AL_temp_GNSS_reg']['p_value']:.2f}",
             size=12, transform=ax5.transAxes)
    ax5.text(0.01, 1.05, '(f)', size=12, transform=ax5.transAxes)
    ax5.set_xlabel("Annual max AL temp. (˚C)", size=14)

    # --------------------------
    # (f) ALT max date vs GNSS vel
    # --------------------------
    ax6.scatter(corr_data['max_ALT_date'],
                corr_data['annual_max_gnss_displacement'] * 1000, color='k')

    ax6.plot(corr_data['max_ALT_date'],
             corr_data['ALT_max_date_GNSS_reg']['slope']
             * corr_data['max_ALT_date']
             + corr_data['ALT_max_date_GNSS_reg']['intercept'], color='k')

    ax6.text(0.05, 0.85, f"R$^2$ = {corr_data['r2_ALT_max_date_GNSS']}",
             size=12, transform=ax6.transAxes)
    ax6.text(0.05, 0.75,
             f"p-value = {corr_data['ALT_max_date_GNSS_reg']['p_value']:.2f}",
             size=12, transform=ax6.transAxes)
    ax6.text(0.01, 1.05, '(e)', size=12, transform=ax6.transAxes)
    ax6.set_xlabel("Date of max ALT (DOY)", size=14)

    plt.subplots_adjust(left=0.1, right=0.93, top=0.9, bottom=0.1)
    
    if Save and outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')

def plot_cumulative_surface_displacement(GNSS_data, SAA_vel, geodetic_data,
                                         SAA_key='tot_monthly_16',
                                         figsize=(10, 6), Save = False,
                                         outpath = None):
    """
    Plot cumulative GNSS, corrected GNSS, SAA surface displacement, and 
    geodetic step-integrated displacement, including summary bars.

    Parameters
    ----------
    GNSS_data : dict
        Must contain:
            'gnss_diff_monthly_all_16_23'
            'gnss_diff_monthly_23_incli_merge'
    
    SAA_vel : dict
        Must contain 'tot_monthly_16' (default key) 
        with index for time and 'surf_cum_defo'.

    geodetic_data : dict
        Must contain 'geodetic_vel_data_nearest_17_23'
        with 'survey_date' and 'vel2d'.

    SAA_key : str
        The dictionary key for SAA surface displacement.

    figsize : tuple
        Size of the figure.
    """

    # ------------------------------------------------
    # Prepare geodetic cumulative displacement
    # ------------------------------------------------
    geo = geodetic_data['geodetic_vel_data_nearest_17_23'].copy()
    geo['survey_date'] = pd.to_datetime(geo['survey_date'])

    # Add reference point in 2016
    start_point_16 = pd.DataFrame({
        'survey_date': [pd.Timestamp('2016-09-01')],
        'vel2d': [0]
    })

    surf_vel = pd.concat([start_point_16, geo]).sort_values('survey_date')
    survey_dates = surf_vel['survey_date']
    survey_deformation = surf_vel['vel2d'].cumsum() * 100  # cm

    # ------------------------------------------------
    # Extract final cumulative values
    # ------------------------------------------------
    final_cum_gnss_monthly = int(
        GNSS_data['gnss_diff_monthly_all_16_23']['horiz_displ_filt_cum'].iloc[-1] * 100
    )
    final_cum_gnss_monthly_corr = int(
        GNSS_data['gnss_diff_monthly_23_incli_merge']['horiz_displ_filt_cum_corr'].iloc[-1]
    )
    final_cum_SAA_surf = int(SAA_vel[SAA_key]['surf_cum_defo'].iloc[-1] / 10)
    final_cum_geodetic = int(survey_deformation.iloc[-1])

    # ------------------------------------------------
    # Create figure
    # ------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            gridspec_kw={'width_ratios': [4, 1]})

    # ------------------------------------------------
    # (Left) Time series cumulative displacement
    # ------------------------------------------------

    # GNSS
    axs[0].plot(
        GNSS_data['gnss_diff_monthly_all_16_23']['time'],
        GNSS_data['gnss_diff_monthly_all_16_23']['horiz_displ_filt_cum'] * 100,
        label='GNSS', color='k', linewidth=2.5
    )

    # GNSS corrected
    axs[0].plot(
        GNSS_data['gnss_diff_monthly_23_incli_merge']['time_x'],
        GNSS_data['gnss_diff_monthly_23_incli_merge']['horiz_displ_filt_cum_corr'],
        label='GNSS corrected', color='#707070', linewidth=2.5
    )

    # SAA
    axs[0].plot(
        SAA_vel[SAA_key].index,
        SAA_vel[SAA_key]['surf_cum_defo'] / 10,
        label='SAA at 0 m', color='#F27202', linewidth=2.5
    )

    # Geodetic
    axs[0].step(
        survey_dates, survey_deformation,
        where='post', label='Geodetic', color='#0852c9', linewidth=2.5
    )

    axs[0].legend(fontsize=14)
    axs[0].minorticks_on()

    axs[0].grid(True, color='grey', alpha=0.3, linestyle='--')
    axs[0].grid(True, which='minor', linestyle=':', color='gray', alpha=0.1)

    # Final numbers on the plot
    axs[0].text(0.83, 0.96, f"{final_cum_gnss_monthly}", transform=axs[0].transAxes)
    axs[0].text(0.93, 0.92, f"{final_cum_gnss_monthly_corr}",
                color='#707070', transform=axs[0].transAxes)
    axs[0].text(0.96, 0.79, f"{final_cum_SAA_surf}",
                color="#F27202", transform=axs[0].transAxes)
    axs[0].text(0.96, 0.84, f"{final_cum_geodetic}",
                color="#0852c9", transform=axs[0].transAxes)

    axs[0].text(0.5, 1.03, "Integrated from monthly or annual velocities",
                ha='center', transform=axs[0].transAxes, size=13)

    axs[0].set_ylabel("Surface displacement (cm)", fontsize=16)
    axs[0].set_xlabel("Year", fontsize=14)
    axs[0].tick_params(labelsize=13)
    axs[0].set_ylim(0, 105)

    # ------------------------------------------------
    # (Right) Total displacement bars
    # ------------------------------------------------
    bar_vals = [final_cum_geodetic, final_cum_gnss_monthly,
                final_cum_gnss_monthly_corr]
    bar_colors = ['#0852c9', 'k', '#707070']

    bars = axs[1].bar(range(len(bar_vals)), bar_vals, color=bar_colors)

    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")

    axs[1].set_xticks([])
    axs[1].set_xticklabels([])
    axs[1].tick_params(labelsize=13)
    axs[1].set_ylim(0, 105)

    for bar in bars:
        height = bar.get_height()
        axs[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.0f}",
            ha='center', va='bottom', fontsize=9
        )

    axs[1].text(1, 107, "Total displacement", ha='center', fontsize=13)

    plt.subplots_adjust(wspace=0)

    if Save and outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')


#%% EXAMPLE PLOTTING

#%% Deformation profiles (Figure 3 in manuscript)

plot_vertical_profile_single_months(SAA_vel['tot_monthly']/10, SAA_data['z_monthly_23'], 
                                    SAA_layers['strain_rates_times'], 87, 
                                    start_date = "2016-06-30", end_date = "2023-08-31", 
                                    min_x = -0.5, max_x = 88, defoSwitch= True, 
                                    velSwitch = False, strainSwitch = False,
                                    save = False)

#%% Printing values shown in Table 1 in manuscript

# Percentage of layer-specific deformation relative to surface velocity

print("Fraction of total displacement at the surface in the AL (%)", SAA_layers['perc_layer_defo']['AL'])
print("Fraction of total displacement at the surface in the ice core (%)", SAA_layers['perc_layer_defo']['core'])
print("Fraction of total displacement at the surface in the shear zone (%)", SAA_layers['perc_layer_defo']['shear'])

# Layer-specific mean velocity (cm/year)

print("AL-specific deformation rate (cm/year)", SAA_layers['layer_mean_vel']["AL"])
print("Ice-core-specific deformation rate (cm/year)", SAA_layers['layer_mean_vel']["core"])
print("Shear-zone-specific deformation rate (cm/year)", SAA_layers['layer_mean_vel']["shear"])

# Layer-specific strain rates (year^-1)

print("AL-specific annual strain rate (year^-1)", SAA_layers['strain_rates']["AL"])
print("Ice-core-specific annual strain rate (year^-1)", SAA_layers['strain_rates']["core"])
print("Shear-zone-specific annual strain rate (year^-1)", SAA_layers['strain_rates']["shear"])

#%% Annual deformation and temperature profiles (Figure 4)

fig, axs = plot_annual_defo_temp_profiles(
    vel_yearly=SAA_vel['vel_yearly'].iloc[:,:-1], # remove last 'time' column
    z_daily=SAA_data['z_daily_23'],
    temp_profiles=temp_data['temp_15_cor_annual'].iloc[1:-1, :-1], # select only 2017-22 (full years) and remove last 'hydro_years' column
    temp_depths=temp_data['depths_15_cor'],
    years=['2017','2018','2019','2020','2021','2022'],
    save=False, outpath="annual_defo_temp_profiles.png"
)

#%% Layer-specific annual and monthly velocity (Figure 5 in manuscript)

plot_defo_layers(GNSS_data['gnss_diff_yearly'], geodetic_data['plot_geodetic_vel_point'],
                 SAA_layers['layer_yearly'], GNSS_data['gnss_diff_monthly_23'],
                 SAA_layers['layer_monthly'], 
                 save_path = 'defo_annual_monthly_layers_gnss_geo_comp.png')

#%% Environmental controls with all deformation complete plot (Figure 6)

plot_ro3_daily_ZCsbars_Qg_Qm_vel(temp_data['temp_15_cor_daily'], temp_data['temp_15_cor_layers'], 
                           meteo_data['daily_means_extended'], 
                           meteo_data['early_winter_daily'], meteo_data['rainfall_extrapolated'], 
                           temp_data['spring_ZCs_15'], temp_data['ground_heat_flux_15'], 
                           meteo_data['snowmelt_nrg'],
                           SAA_layers['layer_monthly'], GNSS_data['gnss_diff_monthly_23'], meteo_data['snow_end_time_16'],
                           temp_data['freezing_onset_time_15'], False)

#%% Spring melt dynamics and kinematics - subplots

plot_spring_melt_deformation_corrs(corr_data, Save = False, outpath = None)

#%% Cumulative surface displacement for 3 different measurement systems (integrated and total displacement)

plot_cumulative_surface_displacement(GNSS_data, SAA_vel, geodetic_data,
                                     Save = False, outpath = None)