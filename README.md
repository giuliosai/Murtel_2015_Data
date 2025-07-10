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

## GNSS displacaement data

The GNSS surface displacement data from 2016 to 2023 is provided by Jan Beutel and the PermaSense project. It is measured continuously and processed at a daily resolution. 

## Geodetic annual displacement data

The geodetic annual surface displacement data from 2009 to present can be accessed from the PERMOS data portal ("[Rock glacier velocity](https://www.permos.ch/data-portal/rock-glacier-velocities)"). Measured annual typically in late August. In this study the marker on the same boulder as the GNSS station (one also closest to borehole) is the only one used. 

## Borehole temperature data

The temperature records from the 1987 and 2015 boreholes are provided in this repository. This data can also be accessed via the PERMOS data portal ("[Permafrost temperature and ALT](https://www.permos.ch/data-portal/permafrost-temperature-and-active-layer)"). 

**Instructions:** Run the script _import_CORtemp.py_ to import the borehole temperature data from both boreholes, and fill data gaps.

## Meteorological data

The meteorological data comes from the _in situ_ weather station installed by [PERMOS](https://www.permos.ch/data-portal/meteo-data) in 1997. The data can also be requested via PERMOS directly. It measures a long list of meteorological variables, however, for the purpose of this study only a subset is kept. The MeteoSwiss station at Piz Corvatsch is also used for the precipitation and temperature data. This data can be requested by contacting MeteoSwiss. The PERMA-XT dataset used for its _in situ_ precipitation measurements can be accessed through [PERMOS](https://www.permos.ch/doi/permos-spec-2023-1). 

**Instructions:** Run the script _import_meteo.py_ to import the meteorological data. 
