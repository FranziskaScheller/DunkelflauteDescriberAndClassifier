"""
This script executes the code. It contains the following modules:
- ETL
-
"""
import pandas as pd
import numpy as np
import cdsapi
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import config
import ETL
import Preprocessor
import pickle
import json

if config.ETL:

    if config.ETL_energy_vars_API:

        ETL.EnergyVarsLoaderAPI()

    if config.ETL_energy_vars_Load:

        solar_pv_power_CFR, solar_pv_power_NRG, wind_power_offs_CFR, wind_power_offs_NRG, wind_power_ons_CFR, wind_power_ons_NRG = ETL.EnergyVarsLoaderFromCSV()

    if config.ETL_meteo_vars_API:

        ETL.MeterologyVarsLoaderAPIManually()


#todo: check if sum of capacity of variables < threshold is correct or each of them needs to be
"""
SPV capacity factor (CFR) is defined as the ratio of actual generation over installed capacity 
(sum of the peak capacity of all PV systems installed in the region of interest). 
The solar PV capacity factor is calculated at grid point level. 
It is important to highlight that this quantity does not represent the power production of a single PV system. 
Instead, it is designed to represent the aggregated production of the PV plant installed in each pixel.
"""
"""
Wind Power capacity factor (CFR), defined as the ratio of actual generation over installed capacity, 
is calculated at grid point level, considering one single wind turbine type for onshore wind, 
and one for offshore wind. It is assumed that one turbine is located at each grid point, 
the turbine type depending only on the grid point type (land or ocean). 
All turbines are assumed to have a hub height of 100 m. This was not meant to faithfully represent 
wind farms locations and characteristics, but rather to illustrate the use of C3S indicators. 
As a consequence, the estimated capacity factors are generally overestimated compared to observed ones as 
i) the real turbines installed have various characteristics, including lower hub height, lower installed capacity, etc. 
and ii) our estimation does not take into account turbines’ unavailability for maintenance or failures.
"""

"""
Calculate moving averages
"""
if config.Preprocessor:

    CFR_sum_solar_wind = Preprocessor.CFR_Aggregator(solar_pv_power_CFR, wind_power_ons_CFR, wind_power_offs_CFR)
    if config.Preprocessor_calc_mov_avg:
        # todo: check if we want to include rows where all entries for all variables are zero
        solar_pv_wind_power_moving_avg = Preprocessor.MovingAveragesCalculator(CFR_sum_solar_wind)
        # solar_pv_wind_power_moving_avg.to_csv(config.file_path + 'solar_pv_wind_power_moving_avg.csv')

    solar_pv_wind_power_moving_avg = pd.read_csv(config.file_path + 'solar_pv_wind_power_moving_avg.csv', index_col= False)
    solar_pv_wind_power_moving_avg = solar_pv_wind_power_moving_avg[solar_pv_wind_power_moving_avg.columns[1:]]

    installed_capacity_solar_pv_power = Preprocessor.InstalledCapacityCorrector(solar_pv_wind_power_moving_avg, CFR_sum_solar_wind)
    print(1)
    solar_pv_power_CFR[solar_pv_power_CFR.columns[1:]] = solar_pv_power_CFR[solar_pv_power_CFR.columns[1:]].round(3)
    # solar_pv_power_CFR[solar_pv_power_CFR['Date'].dt.year <= 1985].to_csv(
    #     '/Volumes/PortableSSD/test.csv', index=False)
    solar_pv_power_CFR[solar_pv_power_CFR['Date'].dt.year <= 2022].to_csv(
       '/Volumes/PortableSSD/test.csv', sep=';', encoding = 'latin1')
    #todo: try low memory thing or encoding= 'unicode_escape'

    # installed_capacity_solar_pv_power[installed_capacity_solar_pv_power['Date'].dt.year <= 1979].to_json(
    #     path_or_buf='/Volumes/PortableSSD/test.pkl', orient="split")
    print(2)
    #df1 = pd.read_json(path_or_buf='/Volumes/PortableSSD/test.csv', orient="split")
    test = pd.read_csv('/Volumes/PortableSSD/test.csv', error_bad_lines=False, sep=';', encoding = 'latin1')
    print(3)
    #installed_capacity_solar_pv_power = solar_pv_power_NRG.drop(columns = 'Date').div(solar_pv_power_CFR.drop(columns = 'Date'))

    dunkelflaute_date_list = Preprocessor.HistDunkelflauteDetector(installed_capacity_solar_pv_power)


# sum of capacities below ...% (?)

#ETL.FileDownloadInsights(config.file_path_energy_vars, config.file_names_energy_vars)



import netCDF4 as nc
fn = '/Users/franziska/PycharmProjects/DunkelflauteDescriberAndClassifier/download/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S197901010000_E197901312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
ds = nc.Dataset(fn)

time = ds['time'][:]
longitude = ds['longitude'][:]
latitude = ds['latitude'][:]
ssrd = ds['ssrd'][:]

name = list(ds.variables.values())[3].name

import numpy as np

ssrd_data = ssrd.filled(np.NaN)

unit = ds.variables['time'].units

import datetime

ref_date = datetime.datetime(int(unit[12:16]), int(unit[17:19]), int(unit[20:22]))
start_date = ref_date + datetime.timedelta(hours = int(ds.variables['time'][0]))

dates = pd.Series(start_date)
for d in range(1,len(time)):
    dates = dates.append(pd.Series(ref_date + datetime.timedelta(hours = int(ds.variables['time'][d]))))


file_names = ['H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S197901010000_E197901312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc','H_ERA5_ECMW_T639_MSL_0000m_Euro_025d_S197901010000_E197901312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc','H_ERA5_ECMW_T639_MSL_0000m_Euro_025d_S197901010000_E197912312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc','H_ERA5_ECMW_T639_SPV_0000m_Euro_025d_S197901010000_E197912312300_CFR_MAP_01h_NA-_noc_org_NA_NA---_NA---_PhM02.nc','H_ERA5_ECMW_T639_TA-_0002m_Euro_025d_S197901010000_E197912312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc','H_ERA5_ECMW_T639_TP-_0000m_Euro_025d_S197901010700_E197912312300_ACC_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc','H_ERA5_ECMW_T639_WOF_0100m_Euro_025d_S197901010000_E197912312300_CFR_MAP_01h_NA-_noc_org_NA_NA---_NA---_PhM01.nc','H_ERA5_ECMW_T639_WON_0100m_Euro_025d_S197901010000_E197912312300_CFR_MAP_01h_NA-_noc_org_NA_NA---_NA---_PhM01.nc','H_ERA5_ECMW_T639_WS-_0010m_Euro_025d_S197901010000_E197912312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc','H_ERA5_ECMW_T639_WS-_0100m_Euro_025d_S197901010000_E197912312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc']

path = '/Users/franziska/PycharmProjects/DunkelflauteDescriberAndClassifier/download/'

for f in file_names:
    print(list(ds.variables.values())[3].name)

for f in file_names:
    ds = nc.Dataset(path + f)
    print(ds)
    print(list(ds.variables.values())[3].name)


#todo: write function that adds all variables up in