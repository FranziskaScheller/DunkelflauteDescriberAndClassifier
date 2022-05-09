"""
This script executes the code. It contains the following modules:
- ETL
- Preprocessor
- Dunkelflaute Describer
- Dunkelflaute Classifier
In the config file it can be specified which part of the code you want to run.
"""
import pandas as pd
import numpy as np
import cdsapi
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import config
import ETL
import Preprocessor
import DFDescriber
import csv
import netCDF4 as nc

if config.ETL:

    if config.ETL_energy_vars_API:

        ETL.EnergyVarsLoaderAPI()

    if config.ETL_energy_vars_Load:

        solar_pv_power_CFR, solar_pv_power_NRG, wind_power_offs_CFR, wind_power_offs_NRG, wind_power_ons_CFR, wind_power_ons_NRG = ETL.EnergyVarsLoaderFromCSV()

    if config.ETL_meteo_vars_API:

        ETL.MeterologyVarsLoaderAPIManually(2019, 2020)

    if config.ETL_meteo_vars_Write:

        data_ssrd_1979, dates_ssrd_1979 = ETL.MeterologyVarsLoaderGHI(['1979'])

        data_msl_1979, dates_msl_1979 = ETL.MeterologyVarsLoader(['1979'], ['MSL_0000m', 'msl'])
        data_ta_1979, dates_ta_1979 = ETL.MeterologyVarsLoader(['1979'], ['TA-_0002m', 't2m'])
        data_ws10_1979, dates_ws10_1979 = ETL.MeterologyVarsLoader(['1979'], ['WS-_0010m', 'ws10'])
        data_ws100_1979, dates_ws100_1979 = ETL.MeterologyVarsLoader(['1979'], ['WS-_0100m', 'var_100_metre_wind_speed'])

    if config.ETL_meteo_vars_Reader:

        data_ssrd_1979, dates_ssrd_1979 = ETL.MeterologyVarsReaderGHI(['1979'])
        data_msl_1979, dates_msl_1979 = ETL.MeterologyVarsReader(['1979'], ['MSL_0000m', 'msl'])
        data_ta_1979, dates_ta_1979 = ETL.MeterologyVarsReader(['1979'], ['TA-_0002m', 't2m'])
        data_ws10_1979, dates_ws10_1979 = ETL.MeterologyVarsReader(['1979'], ['WS-_0010m', 'ws10'])
        data_ws100_1979, dates_ws100_1979 = ETL.MeterologyVarsReader(['1979'], ['WS-_0100m', 'var_100_metre_wind_speed'])

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
First, we need to determine on which days a Dunkelflaute event occured. We orient the Definition of a Dunkelflaute event 
on the paper "A Brief Climatology of Dunkelflaute Events over and Surrounding the North and Baltic Sea Areas", where 
it is defined as the solar capacity factor AND wind capacity factor both falling under 20% for at least 24 hours in a row.

Since for example solar power is known to be dependent on seasonality, we first "usual capacity factors" by using a 
rolling window approach. Therefore, we calculate the "usual capacity factor" for each hour of each day by taking the 
data of the x (specified in config) hours before and after this date as well as the data of the same days and months but 
over all other available years to calculate the average value of the solar or wind capacity factor. 
Afterwards we divide each entry of the original capacity factor by the corresponding entry in the usual capacity factor data 
frame and use this data to check whether the entries lie below the 20% threshold for at least 24 hours in a row. 
If this is true for the same dates in all capacity factor data sets (solar AND wind) we spcify this as the Dunkelflaute event.
"""

"""
 Calculate rolling window averages for the capacity factor data 
"""
if config.Preprocessor:

    #CFR_sum_solar_wind = Preprocessor.CFR_Aggregator(solar_pv_power_CFR, wind_power_ons_CFR, wind_power_offs_CFR)
    if config.Preprocessor_calc_mov_avg:
        # todo: check if we want to include rows where all entries for all variables are zero
        solar_pv_power_CFR_moving_avg = Preprocessor.MovingAveragesCalculator(solar_pv_power_CFR)
        solar_pv_power_CFR_moving_avg.to_csv(
            config.file_path_ext_ssd + 'solar_pv_power_CFR_moving_avg.csv', sep=';', encoding='latin1', index=False)

        wind_power_ons_moving_avg = Preprocessor.MovingAveragesCalculator(wind_power_ons_CFR)
        wind_power_ons_moving_avg.to_csv(
            config.file_path_ext_ssd + 'wind_power_ons_moving_avg.csv', sep=';', encoding='latin1', index=False)

        wind_power_offs_moving_avg = Preprocessor.MovingAveragesCalculator(wind_power_offs_CFR)
        wind_power_offs_moving_avg.to_csv(
            config.file_path_ext_ssd + 'wind_power_offs_moving_avg.csv', sep=';', encoding='latin1', index=False)
        # solar_pv_wind_power_moving_avg.to_csv(config.file_path + 'solar_pv_wind_power_moving_avg.csv')

    if config.Preprocessor_read_data_mov_avg:
        solar_pv_power_CFR_moving_avg = pd.read_csv(config.file_path_ext_ssd + 'solar_pv_power_CFR_moving_avg.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)
        wind_power_ons_moving_avg = pd.read_csv(config.file_path_ext_ssd + 'wind_power_ons_moving_avg.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)
        wind_power_offs_moving_avg = pd.read_csv(config.file_path_ext_ssd + 'wind_power_offs_moving_avg.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)

        solar_pv_power_CFR_moving_avg['Date'] = solar_pv_power_CFR_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_ons_moving_avg['Date'] = wind_power_ons_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_offs_moving_avg['Date'] = wind_power_offs_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # for 29th of February fill in the mean of the 28th of February and 01st of March
    for h in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']:
        mean_val_solar_pv = np.nanmean(solar_pv_power_CFR_moving_avg[solar_pv_power_CFR_moving_avg['Date'].isin(
            [datetime.strptime('1979-02-28 ' + h + ':00:00', "%Y-%m-%d %H:%M:%S"),
            datetime.strptime('1979-03-01 ' + h + ':00:00', "%Y-%m-%d %H:%M:%S")])].drop(columns='Date'), axis=0)
        mean_val_wind_ons = np.nanmean(wind_power_ons_moving_avg[wind_power_ons_moving_avg['Date'].isin(
            [datetime.strptime('1979-02-28 ' + h + ':00:00', "%Y-%m-%d %H:%M:%S"),
             datetime.strptime('1979-03-01 ' + h + ':00:00', "%Y-%m-%d %H:%M:%S")])].drop(columns='Date'), axis=0)
        mean_val_wind_offs = np.nanmean(wind_power_offs_moving_avg[wind_power_offs_moving_avg['Date'].isin(
            [datetime.strptime('1979-02-28 ' + h + ':00:00', "%Y-%m-%d %H:%M:%S"),
             datetime.strptime('1979-03-01 ' + h + ':00:00', "%Y-%m-%d %H:%M:%S")])].drop(columns='Date'), axis=0)
        dates_29_feb = solar_pv_power_CFR_moving_avg['Date'][
            (solar_pv_power_CFR_moving_avg['Date'].apply(lambda x: x.day) == 29) & (
                        solar_pv_power_CFR_moving_avg['Date'].apply(lambda x: x.month) == 2) & (
                        solar_pv_power_CFR_moving_avg['Date'].apply(lambda x: x.hour) == int(h))]
        for date_i in dates_29_feb:
            solar_pv_power_CFR_moving_avg.loc[solar_pv_power_CFR_moving_avg['Date'] == date_i, solar_pv_power_CFR_moving_avg.columns[1:]] = mean_val_solar_pv.T
            wind_power_ons_moving_avg.loc[wind_power_ons_moving_avg['Date'] == date_i, wind_power_ons_moving_avg.columns[1:]] = mean_val_wind_ons.T
            wind_power_offs_moving_avg.loc[wind_power_offs_moving_avg['Date'] == date_i, wind_power_offs_moving_avg.columns[1:]] = mean_val_wind_offs.T

    installed_capacity_factor_solar_pv_power = Preprocessor.InstalledCapacityCorrector(solar_pv_power_CFR_moving_avg, solar_pv_power_CFR)
    installed_capacity_factor_wind_power_ons = Preprocessor.InstalledCapacityCorrector(wind_power_ons_moving_avg,
                                                                                 wind_power_ons_CFR)
    installed_capacity_factor_wind_power_offs = Preprocessor.InstalledCapacityCorrector(wind_power_offs_moving_avg,
                                                                                wind_power_offs_CFR)

    installed_capacity_factor_solar_pv_power.to_csv(
        config.file_path_ext_ssd + 'installed_capacity_factor_solar_pv_power.csv', sep=';', encoding='latin1', index=False)
    installed_capacity_factor_wind_power_ons.to_csv(
        config.file_path_ext_ssd + 'installed_capacity_factor_wind_power_ons.csv', sep=';', encoding='latin1', index=False)
    installed_capacity_factor_wind_power_offs.to_csv(
        config.file_path_ext_ssd + 'installed_capacity_factor_wind_power_offs.csv', sep=';', encoding='latin1', index=False)

    #dunkelflaute_dates_country_i = Preprocessor.HistDunkelflauteDetector(installed_capacity_solar_pv_power, 'DE')
    dunkelflaute_dates_country_i = Preprocessor.HistDunkelflauteDetector(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE')
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', 0.2)
    #----

    Preprocessor.HistPlotterDunkelflauteEvents(pd.read_csv('DF_relative_counts_per_nbr_of_hours_DE_.csv', sep=';', encoding='latin1'), 'Germany(DE)')

    #dunkelflaute_freq_all_countries  = Preprocessor.HistDunkelflauteDetectorFrequencysAllCountries(installed_capacity_solar_pv_power)

"""
Plot mean of meteo variables for Dunkelflaute events
"""
if config.Describer:

    if config.DescriberMSLDataCalculator:

        DF_Data_all_mean = DFDescriber.MeteoVarssAggregatorForDunkelflauteEvents(installed_capacity_solar_pv_power, ['MSL_0000m', 'msl'], 'DE')

    if config.DescriberMSLPlotter:
        import netCDF4 as nc

        fn = '/Volumes/PortableSSD/download19790102/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S197901010000_E197901312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
        ds = nc.Dataset(fn)

        time = ds['time'][:]
        longitude = ds['longitude'][:]
        latitude = ds['latitude'][:]

        DFDescriber.MeteoVarsPlotter(dunkelflaute_dates_country_i, data_msl_1979, dates_msl_1979, 'msl', longitude, latitude)


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