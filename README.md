# Murtèl Data

The dataset presented here is accompanying the publication _"Multi-annual and seasonal patterns of Murtèl rock glacier borehole deformation, environmental controls and implications for kinematic monitoring"_ submitted to The Cryosphere.

## Get started
1. Clone the repository:
   ```bash
   git clone https://github.com/giuliosai/Murtel_2015_Data.git
   ```
2. In your code viewer, set the working directory to the Murtel_2015_Data folder now created on your disk.
3. Run _process_SAA_data.py_ and _process_other_data.py_ to define all functions to import and process all necessary data.
4. Run first cell of _murtel_plotting.py_ to import all the data. Later cells can be run individually to plot the desired figure. 

## Borehole deformation data (SAA)

The inclinometer (SAA) data consists of MATLAB files which are the output from the Mesurand software which converts raw data (angles) to useful position data. The data is split into 2016-17, 2017-18 and 2018-23 periods. Note that the one for the 2018-2023 period is missing as it is too large, but can be found in the [Zenodo data publication](https://doi.org/10.5281/zenodo.15782681).

**Instructions:** Make sure you have added the missing _multi_saa_allcart_18_23.mat_ file in the SAA_data subfolder (see Zenodo dataset). Run the script _process_SAA_data.py_ to prepare the functions for importing the SAA data, extracting variables out of the MATLAB files, cleaning the data and exporting into a useful CSV format. The script also contains procedures developed to resample the data to various temporal resolutions, calculate the vertical strain rate, calculate layer-specific deformation both as absolute and percentage values.

## GNSS displacaement data

The GNSS surface displacement data from 2016 to 2023 is provided by Jan Beutel and the PermaSense project. It is measured continuously and processed at a daily resolution. The functions to import and process it are found under the _process_other_data.py_.

## Geodetic annual displacement data

The geodetic annual surface displacement data from 2009 to present can be accessed from the PERMOS data portal ("[Rock glacier velocity](https://www.permos.ch/data-portal/rock-glacier-velocities)"). Measured annual typically in late August. In this study we only used the COR_021 marker located on the same boulder as the GNSS station and closest to the 2015 borehole. The functions to import and process the geodetic data are found under the _process_other_data.py_.

## Borehole temperature data

The temperature records from the 1987 and 2015 boreholes are provided in this repository. This data can also be accessed via the PERMOS data portal ("[Permafrost temperature and ALT](https://www.permos.ch/data-portal/permafrost-temperature-and-active-layer)"). The functions to import and process the borehole temperature data are found under the _process_other_data.py_.

## Meteorological data

The meteorological data comes from the _in situ_ weather station installed by [PERMOS](https://www.permos.ch/data-portal/meteo-data) in 1997. The data can also be requested via PERMOS directly. It measures a long list of meteorological variables, however, for the purpose of this study only a subset is kept. The MeteoSwiss station at Piz Corvatsch is also used for the precipitation and temperature data. This data can be requested by contacting MeteoSwiss. The PERMA-XT dataset used for its _in situ_ precipitation measurements can be accessed through [PERMOS](https://www.permos.ch/doi/permos-spec-2023-1). The functions to import and process the meteo data are found under the _process_other_data.py_.

## Visualizing the data (generating plots from the paper)

Open the _murtel_plotting.py_ script. The working directory should be set to the Murtel_2015_Data_Git folder with the corresponding subfolders with the data files in them. Make sure you have already ran the _process_SAA_data.py_ and the _process_other_data.py_ scripts, and then run the first cell of _murtel_plotting.py_ to load the data and define the plotting functions. The latter cells in the script generate the individual plots. 
