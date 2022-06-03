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

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'sis-energy-derived-reanalysis',
    {
        'variable': 'wind_power_generation_onshore',
        'spatial_aggregation': 'sub_country_level',
        'energy_product_type': 'capacity_factor_ratio',
        'temporal_aggregation': 'hourly',
        'format': 'zip',
    },
    'downloadnuts02.zip')
print(1)
# import cdsapi
#
# c = cdsapi.Client()
#
# c.retrieve(
#     'sis-energy-derived-reanalysis',
#     {
#         'format': 'zip',
#         'variable': [
#             'solar_photovoltaic_power_generation', 'wind_power_generation_offshore', 'wind_power_generation_onshore',
#         ],
#         'spatial_aggregation': [
#             'country_level', 'maritime_country_level',
#         ],
#         'energy_product_type': [
#             'energy', 'power',
#         ],
#         'temporal_aggregation': 'annual',
#     },
#     'download_energy_power_annual.zip')

if config.ETL:

    if config.ETL_energy_vars_API:

        ETL.EnergyVarsLoaderAPI()

    if config.ETL_energy_vars_Load:

        solar_pv_power_CFR, solar_pv_power_NRG, wind_power_offs_CFR, wind_power_offs_NRG, wind_power_ons_CFR, wind_power_ons_NRG = ETL.EnergyVarsLoaderFromCSV()

        # Note: zeros are just empty entries in csv and therefore 'nan', so we fill nan's with zero for solar/PV
        # for wind this problem doesn't occur because then very small values (such as 0.00001) are given
        solar_pv_power_CFR = solar_pv_power_CFR.fillna(0)

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

    #Preprocessor.HistPlotterDunkelflauteEvents(pd.read_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.7_threshold2.csv', sep=';'), 'Germany(DE)')
    #print(1)

    #CFR_sum_solar_wind = Preprocessor.CFR_Aggregator(solar_pv_power_CFR, wind_power_ons_CFR, wind_power_offs_CFR)
    if config.Preprocessor_calc_mov_avg:
        # todo: check if we want to include rows where all entries for all variables are zero
        #solar_pv_power_CFR_moving_avg = Preprocessor.MovingAveragesCalculatorSolarPV(solar_pv_power_CFR)
        #solar_pv_power_CFR_moving_avg.to_csv(
        #    'solar_pv_power_CFR_moving_avg.csv', sep=';', encoding='latin1', index=False)
        #solar_pv_power_CFR_moving_avg_h = Preprocessor.MovingAveragesCalculatorSolarPVHourly(solar_pv_power_CFR)

        solar_pv_power_CFR_moving_avg_h2 = Preprocessor.MovingAveragesCalculatorSolarPVHourly2(solar_pv_power_CFR)

        solar_pv_power_CFR_moving_avg_h2.to_csv(
            'solar_pv_power_CFR_moving_avg_h2.csv', sep=';', encoding='latin1', index=False)

        wind_power_ons_moving_avg = Preprocessor.MovingAveragesCalculator(wind_power_ons_CFR)
        wind_power_ons_moving_avg.to_csv('wind_power_ons_moving_avg.csv', sep=';', encoding='latin1', index=False)
        # wind_power_ons_moving_avg.to_csv(
        #     config.file_path_ext_ssd + 'wind_power_ons_moving_avg.csv', sep=';', encoding='latin1', index=False)

        wind_power_offs_moving_avg = Preprocessor.MovingAveragesCalculator(wind_power_offs_CFR)
        wind_power_offs_moving_avg.to_csv('wind_power_offs_moving_avg.csv', sep=';', encoding='latin1', index=False)
#        wind_power_offs_moving_avg.to_csv(
#            config.file_path_ext_ssd + 'wind_power_offs_moving_avg.csv', sep=';', encoding='latin1', index=False)

        # solar_pv_wind_power_moving_avg.to_csv(config.file_path + 'solar_pv_wind_power_moving_avg.csv')

    if config.Preprocessor_read_data_mov_avg:
        solar_pv_power_CFR_moving_avg = pd.read_csv('solar_pv_power_CFR_moving_avg_h2.csv', error_bad_lines=False,
                                                    sep=';', encoding='latin1', index_col=False)

        wind_power_ons_moving_avg = pd.read_csv('wind_power_ons_moving_avg.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)

        wind_power_offs_moving_avg = pd.read_csv('wind_power_offs_moving_avg.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)

        solar_pv_power_CFR_moving_avg['Date'] = solar_pv_power_CFR_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_ons_moving_avg['Date'] = wind_power_ons_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_offs_moving_avg['Date'] = wind_power_offs_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    if config.Preprocessor_installed_capacity_corrector:
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

        # solar_pv_power_CFR.loc[(
        #                            solar_pv_power_CFR['Date'].apply(lambda x: x.hour).isin(
        #                                [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7])), solar_pv_power_CFR.columns[
        #                                                                                    1:]] = 0

        installed_capacity_factor_solar_pv_power = Preprocessor.InstalledCapacityCorrector(solar_pv_power_CFR_moving_avg, solar_pv_power_CFR)
        installed_capacity_factor_wind_power_ons = Preprocessor.InstalledCapacityCorrector(wind_power_ons_moving_avg,
                                                                                     wind_power_ons_CFR)
        installed_capacity_factor_wind_power_offs = Preprocessor.InstalledCapacityCorrector(wind_power_offs_moving_avg,
                                                                                    wind_power_offs_CFR)

        # installed_capacity_factor_solar_pv_power.loc[(installed_capacity_factor_solar_pv_power['Date'].apply(
        #                                                      lambda x: x.hour).isin(
        #                                                      [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6,
        #                                                       7])), installed_capacity_factor_solar_pv_power.columns[
        #                                                             1:]] = 0
        installed_capacity_factor_solar_pv_power.to_csv(
            'installed_capacity_factor_solar_pv_power_h2.csv', sep=';', encoding='latin1', index=False)
        installed_capacity_factor_wind_power_ons.to_csv(
            'installed_capacity_factor_wind_power_ons.csv', sep=';', encoding='latin1', index=False)
        installed_capacity_factor_wind_power_offs.to_csv(
            'installed_capacity_factor_wind_power_offs.csv', sep=';', encoding='latin1', index=False)

    installed_capacity_factor_solar_pv_power = pd.read_csv('installed_capacity_factor_solar_pv_power_h2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
    installed_capacity_factor_wind_power_ons = pd.read_csv('installed_capacity_factor_wind_power_ons.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
    installed_capacity_factor_wind_power_offs = pd.read_csv('installed_capacity_factor_wind_power_offs.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)

    installed_capacity_factor_solar_pv_power['Date'] = installed_capacity_factor_solar_pv_power['Date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    installed_capacity_factor_wind_power_ons['Date'] = installed_capacity_factor_wind_power_ons['Date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    installed_capacity_factor_wind_power_offs['Date'] = installed_capacity_factor_wind_power_offs['Date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    # installed_capacity_factor_solar_pv_power.loc[(installed_capacity_factor_solar_pv_power['Date'].apply(
    #                                                      lambda x: x.hour).isin(
    #                                                      [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6,
    #                                                       7])), installed_capacity_factor_solar_pv_power.columns[
    #                                                             1:]] = 0
    #dunkelflaute_dates_country_i = Preprocessor.HistDunkelflauteDetector(installed_capacity_solar_pv_power, 'DE')

    #dunkelflaute_dates_country_i = Preprocessor.HistDunkelflauteDetector(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE')

    #dunkelflaute_freq_country_i_th02 = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(installed_capacity_factor_solar_pv_power,
    #'DE', 0.2, 'solar_pv')
    # !!!
    installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power.fillna(0)
    installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power.round(3)
    installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons.round(3)
    installed_capacity_factor_wind_power_offs = installed_capacity_factor_wind_power_offs.round(3)

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'NL', [0.7, 0.7, 0.7])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.6, 0.6, 0.6])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.3, 0.3, 0.3])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'NL', [0.2, 0.2, 0.2])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'NL', [0.5, 0.5, 0.5])

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'ES', [0.7, 0.7, 0.7])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'FR', [0.3, 0.3, 0.3])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'PL', [0.3, 0.3, 0.3])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'ES', [0.3, 0.3, 0.3])

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'PL', [0.7, 0.7, 0.7])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.6, 0.6, 0.6])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.3, 0.3, 0.3])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'PL', [0.2, 0.2, 0.2])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'PL', [0.5, 0.5, 0.5])

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'ES', [0.7, 0.7, 0.7])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.6, 0.6, 0.6])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.3, 0.3, 0.3])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'ES', [0.2, 0.2, 0.2])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'ES', [0.5, 0.5, 0.5])

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.7, 0.7, 0.7])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.6, 0.6, 0.6])
    #dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.3, 0.3, 0.3])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'FR', [0.2, 0.2, 0.2])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'FR', [0.5, 0.5, 0.5])

    #dunkelflaute_freq_country_i_th02 = Preprocessor.FrequencyCalculatorCFRBelowThresholdSolarPVOneThresholds(installed_capacity_factor_solar_pv_power,
    #!!!                                                                                                 'DE', 0.2, 'solar_pv')

    # #dunkelflaute_freq_country_i_th02 = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(installed_capacity_factor_wind_power_ons,
    dunkelflaute_freq_country_i_th02ons = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
        installed_capacity_factor_wind_power_ons, 'DE', 0.2, 'wind_power_onshore')
    dunkelflaute_freq_country_i_th03ons = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
        installed_capacity_factor_wind_power_ons, 'DE', 0.3, 'wind_power_onshore')
    dunkelflaute_freq_country_i_th05ons = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
        installed_capacity_factor_wind_power_ons, 'DE', 0.5, 'wind_power_onshore')
    dunkelflaute_freq_country_i_th02offs = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
        installed_capacity_factor_wind_power_offs, 'DE', 0.2, 'wind_power_offshore')
    dunkelflaute_freq_country_i_th03offs = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
        installed_capacity_factor_wind_power_offs, 'DE', 0.3, 'wind_power_offshore')
    dunkelflaute_freq_country_i_th05offs = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
        installed_capacity_factor_wind_power_offs, 'DE', 0.5, 'wind_power_offshore')
    print(1)
    if config.calc_summary_measures:
        installed_capacity_factor_solar_pv_power['Hour'] = pd.DatetimeIndex(
            installed_capacity_factor_solar_pv_power['Date']).hour
        hourly_mean_med_var_solar_pv = installed_capacity_factor_solar_pv_power[['Hour', 'DE']].groupby(
            ['Hour']).agg(["mean", "median", "var"]).reset_index()
        hourly_mean_med_var_solar_pv.to_csv(
            config.file_path_ext_ssd + 'hourly_mean_med_var_solar_pv.csv', sep=';', encoding='latin1', index=False)

        installed_capacity_factor_wind_power_ons['Hour'] = pd.DatetimeIndex(
            installed_capacity_factor_wind_power_ons['Date']).hour
        hourly_mean_med_var_wind_ons = installed_capacity_factor_wind_power_ons[['Hour', 'DE']].groupby(
            ['Hour']).agg(["mean", "median", "var"]).reset_index()
        hourly_mean_med_var_wind_ons.to_csv(
            config.file_path_ext_ssd + 'hourly_mean_med_var_wind_ons.csv', sep=';', encoding='latin1', index=False)

        installed_capacity_factor_wind_power_offs['Hour'] = pd.DatetimeIndex(
            installed_capacity_factor_wind_power_offs['Date']).hour
        hourly_mean_med_var_wind_offs = installed_capacity_factor_wind_power_offs[['Hour', 'DE']].groupby(
            ['Hour']).agg(["mean", "median", "var"]).reset_index()
        hourly_mean_med_var_wind_offs.to_csv(
            config.file_path_ext_ssd + 'hourly_mean_med_var_wind_offs.csv', sep=';', encoding='latin1', index=False)



    # dunkelflaute_freq_country_i_th01 = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(installed_capacity_factor_solar_pv_power,
    #                                                                                                 'DE', 0.1, 'solar_pv')
    # dunkelflaute_freq_country_i_th03 = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(installed_capacity_factor_solar_pv_power,
    #                                                                                                 'DE', 0.3, 'solar_pv')
    #
    # dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableSeveralThresholds(installed_capacity_factor_solar_pv_power,
    #                                                                                                 'DE', [0.1, 0.2, 0.3], 'solar_pv')

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.3, 0.3, 0.3])


    dunkelflaute_freq_country_i_several_thresholds = Preprocessor.FrequencyCalculatorCFRBelowSeveralThresholds(
        installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons,
        installed_capacity_factor_wind_power_offs, 'DE', [0.05, 0.1, 0.2])

    Preprocessor.HistPlotterDunkelflauteEvents(pd.read_csv('DF_relative_counts_per_nbr_of_hours_DE_.csv', sep=';', encoding='latin1'), 'Germany(DE)')
    # ++++
    DF_relative_counts_per_nbr_of_hours_DE_wind_ons = pd.read_csv(config.file_path_ext_ssd + 'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_ons0.2_threshold.csv', sep=';', encoding='latin1')
    DF_relative_counts_per_nbr_of_hours_DE_wind_offs = pd.read_csv(config.file_path_ext_ssd + 'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_offs0.2_threshold.csv', sep=';', encoding='latin1')
    DF_relative_counts_per_nbr_of_hours_DE_pv_solar = pd.read_csv(config.file_path_ext_ssd + 'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_PV0.2_threshold.csv', sep=';', encoding='latin1')

    dunkelflaute_freq_country_i_several_thresholds = DF_relative_counts_per_nbr_of_hours_DE_pv_solar
    dunkelflaute_freq_country_i_several_thresholds = dunkelflaute_freq_country_i_several_thresholds.merge(DF_relative_counts_per_nbr_of_hours_DE_wind_ons, on = 'LengthsDF')
    Preprocessor.HistPlotterDunkelflauteEventsSeveralThresholdsOneCountry(dunkelflaute_freq_country_i_several_thresholds, 'Germany(DE)', 'solar/PV, wind on- and offshore', '0.2')
    # ++++
    print(1)
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