# Murtèl Data

The dataset presented here is accompanying the publication _"Multi-annual and seasonal patterns of Murtèl rock glacier borehole deformation, environmental controls and implications for kinematic monitoring"_ submitted to The Cryosphere.

## Get started
1. Clone the repository:
   ```bash
   git clone https://github.com/giuliosai/Murtel_2015_Data.git
   ```

## Borehole deformation data (SAA)

The inclinometer (SAA) data consists of MATLAB files which are the output from the Mesurand software which converts raw data (angles) to useful position data. The data is split into 2016-17, 2017-18 and 2018-23 periods. Note that the one for the 2018-2023 period is missing as it is too large to upload to GitHub.

**Instructions:** Run the script _import_SAA.py_ to import the SAA data, extract variables out of the MATLAB files, clean the data and export into a useful CSV format.

## Borehole temperature data

Both the temperature record from the 1987 and 2015 boreholes is available here. Note that the temperature data can also be accessed via the [PERMOS data portal] (https://www.permos.ch/data-portal).  

**Instructions:** Run the script _import_CORtemp.py_ to import the borehole temperature data from both boreholes, and fill data gaps.

## Meteorological data

The meteorological data comes from the _in situ_ weather station installed by [PERMOS](https://www.permos.ch/data-portal/meteo-data) in 1997. It measures a long list of meteorological variables, however, for the purpose of this study only a subset is kept. The MeteoSwiss station at Piz Corvatsch is also used for the precipitation and temperature data. 

**Instructions:** Run the script _import_meteo.py_ to import the meteorological data. 
