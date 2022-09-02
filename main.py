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
from sklearn.linear_model import LinearRegression

# import cdsapi
#
# c = cdsapi.Client()
#
# c.retrieve(
#     'sis-energy-derived-reanalysis',
#     {
#         'variable': 'wind_power_generation_offshore',
#         'spatial_aggregation': 'maritime_sub_country_level',
#         'energy_product_type': 'capacity_factor_ratio',
#         'temporal_aggregation': 'hourly',
#         'format': 'zip',
#     },
#     'download03.zip')
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
        solar_pv_power_CFR = solar_pv_power_CFR[solar_pv_power_CFR['Date'].dt.year <= 2021]

    if config.ETL_meteo_vars_API:

        ETL.MeterologyVarsLoaderAPIManually(1996, 1997)
        print(1)
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


    if config.ETL_RegressionCorrection:

        wind_ons_CFR_nuts02 = pd.read_csv(
            'H_ERA5_ECMW_T639_WON_0100m_Euro_NUT2_S197901010000_E202204302300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
            skiprows=lambda x: x in range(0, 52))
        wind_ons_CFR_nuts02 = wind_ons_CFR_nuts02.drop_duplicates()
        wind_ons_CFR_nuts02['Date'] = wind_ons_CFR_nuts02['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_ons_CFR_nuts02 = wind_ons_CFR_nuts02[wind_ons_CFR_nuts02['Date'].dt.year <= 2021]
        wind_ons_CFR_nuts02_20_21 = wind_ons_CFR_nuts02[
            (wind_ons_CFR_nuts02['Date'].dt.year <= 2021) & (
                        wind_ons_CFR_nuts02['Date'].dt.year >= 2020)].reset_index().drop(columns='index')

        wind_offs_CFR_nuts02 = pd.read_csv(
            'H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR1_S197901010000_E202205312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
            skiprows=lambda x: x in range(0, 52))
        wind_offs_CFR_nuts02 = wind_offs_CFR_nuts02.drop_duplicates()
        wind_offs_CFR_nuts02['Date'] = wind_offs_CFR_nuts02['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_offs_CFR_nuts02 = wind_offs_CFR_nuts02[wind_offs_CFR_nuts02['Date'].dt.year <= 2021]
        wind_offs_CFR_nuts02_20_21 = wind_offs_CFR_nuts02[
            (wind_offs_CFR_nuts02['Date'].dt.year <= 2021) & (
                        wind_offs_CFR_nuts02['Date'].dt.year >= 2020)].reset_index().drop(columns='index')

        if config.ETL_RegressionCorrection_DE:

            wind_act_gen20DE = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202001010000-202101010000DE.csv',
                                                        sep=',')
            wind_act_gen21DE = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202101010000-202201010000DE.csv',
                                                        sep=',')
            wind_act_genDE = wind_act_gen20DE.append(wind_act_gen21DE)
            wind_act_genDE = wind_act_genDE[
                ['MTU', 'Wind Offshore  - Actual Aggregated [MW]', 'Wind Onshore  - Actual Aggregated [MW]']]

            wind_act_genDE['Time'] = pd.to_datetime(wind_act_genDE['MTU'].str[0:10] + ' ' + wind_act_genDE['MTU'].str[11:16] , format = '%d.%m.%Y %H:%M')
            wind_act_genDE = wind_act_genDE.drop(columns = 'MTU')
            wind_act_genDE['Time'] = pd.Series(wind_act_genDE['Time'].apply(lambda x: x.floor('H')))
            wind_act_genDE_aggr = wind_act_genDE.groupby(by=['Time']).mean().reset_index()
            wind_act_genDE_aggr_max_ons = wind_act_genDE_aggr['Wind Onshore  - Actual Aggregated [MW]'].max()
            wind_act_genDE_aggr_max_offs = wind_act_genDE_aggr['Wind Offshore  - Actual Aggregated [MW]'].max()
            wind_act_genDE_aggr['Wind Onshore  - Actual Aggregated [MW]'] = wind_act_genDE_aggr['Wind Onshore  - Actual Aggregated [MW]']/wind_act_genDE_aggr_max_ons
            wind_act_genDE_aggr['Wind Offshore  - Actual Aggregated [MW]'] = wind_act_genDE_aggr[
                                                                                'Wind Offshore  - Actual Aggregated [MW]'] / wind_act_genDE_aggr_max_offs

            #todo: check if we can aggregate with mean or sum
            y_wind_ons = wind_act_genDE_aggr['Wind Onshore  - Actual Aggregated [MW]']
            X_wind_ons = wind_ons_CFR_nuts02_20_21[wind_ons_CFR_nuts02_20_21.columns[64:102]]
            X_wind_ons = X_wind_ons.dropna(axis = 1, how = 'all')

            X_wind_ons_all = wind_ons_CFR_nuts02[wind_ons_CFR_nuts02.columns[64:102]]
            X_wind_ons_all = X_wind_ons_all.dropna(axis=1, how='all')

            regr_wind_ons = LinearRegression(fit_intercept=False, positive=True).fit(X_wind_ons, y_wind_ons)
            weights_regr_wind_ons = regr_wind_ons.coef_
            #intercept_regr_wind_ons = regr_wind_ons.intercept_

            y_wind_offs = wind_act_genDE_aggr['Wind Offshore  - Actual Aggregated [MW]']
            X_wind_offs = wind_offs_CFR_nuts02_20_21[wind_offs_CFR_nuts02_20_21.columns[7:10]]
            X_wind_offs = X_wind_offs.dropna(axis = 1, how = 'all')

            X_wind_offs_all = wind_offs_CFR_nuts02[wind_offs_CFR_nuts02.columns[7:10]]
            X_wind_offs_all = X_wind_offs_all.dropna(axis = 1, how = 'all')

            regr_wind_offs = LinearRegression(fit_intercept=False,positive=True).fit(X_wind_offs, y_wind_offs)
            weights_regr_wind_offs = regr_wind_offs.coef_
            new_CFRs_onshore_wind = pd.DataFrame(np.dot(X_wind_ons_all, weights_regr_wind_ons), columns = ['DE'])
            new_CFRs_onshore_wind.insert(0, 'Date', wind_ons_CFR_nuts02['Date'].reset_index().drop(columns = 'index'))
            new_CFRs_offshore_wind = pd.DataFrame(np.dot(X_wind_offs_all, weights_regr_wind_offs), columns = ['DE'])
            new_CFRs_offshore_wind.insert(0, 'Date', wind_offs_CFR_nuts02['Date'].reset_index().drop(columns = 'index'))
            new_CFRs_onshore_wind.to_csv(
                'new_CFRs_onshore_wind.csv', sep=';', encoding='latin1', index=False)
            new_CFRs_offshore_wind.to_csv(
                'new_CFRs_offshore_wind.csv', sep=';', encoding='latin1', index=False)

        if config.ETL_RegressionCorrection_NL:
            # NL
            wind_act_gen20NL = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202001010000-202101010000NL.csv',
                                                        sep=',')
            wind_act_gen21NL = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202101010000-202201010000NL.csv',
                                                        sep=',')
            wind_act_genNL = wind_act_gen20NL.append(wind_act_gen21NL)
            wind_act_genNL = wind_act_genNL[
                ['MTU', 'Wind Offshore  - Actual Aggregated [MW]', 'Wind Onshore  - Actual Aggregated [MW]']]

            wind_act_genNL['Time'] = pd.to_datetime(wind_act_genNL['MTU'].str[0:10] + ' ' + wind_act_genNL['MTU'].str[11:16] , format = '%d.%m.%Y %H:%M')
            wind_act_genNL = wind_act_genNL.drop(columns = 'MTU')
            wind_act_genNL['Time'] = pd.Series(wind_act_genNL['Time'].apply(lambda x: x.floor('H')))
            wind_act_genNL_aggr = wind_act_genNL.groupby(by=['Time']).mean().reset_index()
            wind_act_genNL_aggr_max_ons = wind_act_genNL_aggr['Wind Onshore  - Actual Aggregated [MW]'].max()
            wind_act_genNL_aggr['Wind Onshore  - Actual Aggregated [MW]'] = wind_act_genNL_aggr['Wind Onshore  - Actual Aggregated [MW]']/wind_act_genNL_aggr_max_ons

            #todo: check if we can aggregate with mean or sum
            y_wind_ons = wind_act_genNL_aggr['Wind Onshore  - Actual Aggregated [MW]']
            X_wind_ons = wind_ons_CFR_nuts02_20_21[wind_ons_CFR_nuts02_20_21.columns[215:227]]
            X_wind_ons = X_wind_ons.dropna(axis = 1, how = 'all')

            X_wind_ons_all = wind_ons_CFR_nuts02[wind_ons_CFR_nuts02.columns[215:227]]
            X_wind_ons_all = X_wind_ons_all.dropna(axis=1, how='all')

            regr_wind_ons = LinearRegression(fit_intercept=False, positive=True).fit(X_wind_ons, y_wind_ons)
            weights_regr_wind_ons = regr_wind_ons.coef_

            new_CFRs_onshore_windNL = pd.DataFrame(np.dot(X_wind_ons_all, weights_regr_wind_ons), columns = ['NL'])
            new_CFRs_onshore_windNL.insert(0, 'Date', wind_ons_CFR_nuts02['Date'].reset_index().drop(columns = 'index'))
            new_CFRs_onshore_windNL.to_csv(
                'new_CFRs_onshore_windNL.csv', sep=';', encoding='latin1', index=False)

        if config.ETL_RegressionCorrection_PL:
            # PL
            wind_act_gen20PL = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202001010000-202101010000PL.csv',
                                                        sep=',')
            wind_act_gen21PL = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202101010000-202201010000PL.csv',
                                                        sep=',')
            wind_act_genPL = wind_act_gen20PL.append(wind_act_gen21PL)
            wind_act_genPL = wind_act_genPL[
                ['MTU', 'Wind Offshore  - Actual Aggregated [MW]', 'Wind Onshore  - Actual Aggregated [MW]']]

            wind_act_genPL['Time'] = pd.to_datetime(wind_act_genPL['MTU'].str[0:10] + ' ' + wind_act_genPL['MTU'].str[11:16] , format = '%d.%m.%Y %H:%M')
            wind_act_genPL = wind_act_genPL.drop(columns = 'MTU')
            wind_act_genPL['Time'] = pd.Series(wind_act_genPL['Time'].apply(lambda x: x.floor('H')))
            wind_act_genPL_aggr = wind_act_genPL.groupby(by=['Time']).mean().reset_index()
            wind_act_genPL_aggr_max_ons = wind_act_genPL_aggr['Wind Onshore  - Actual Aggregated [MW]'].max()
            wind_act_genPL_aggr['Wind Onshore  - Actual Aggregated [MW]'] = wind_act_genPL_aggr['Wind Onshore  - Actual Aggregated [MW]']/wind_act_genPL_aggr_max_ons

            #todo: check if we can aggregate with mean or sum
            y_wind_ons = wind_act_genPL_aggr['Wind Onshore  - Actual Aggregated [MW]']
            X_wind_ons = wind_ons_CFR_nuts02_20_21[wind_ons_CFR_nuts02_20_21.columns[234:251]]
            X_wind_ons = X_wind_ons.dropna(axis = 1, how = 'all')

            X_wind_ons_all = wind_ons_CFR_nuts02[wind_ons_CFR_nuts02.columns[234:251]]
            X_wind_ons_all = X_wind_ons_all.dropna(axis=1, how='all')

            regr_wind_ons = LinearRegression(fit_intercept=False, positive=True).fit(X_wind_ons, y_wind_ons)
            weights_regr_wind_ons = regr_wind_ons.coef_

            new_CFRs_onshore_windPL = pd.DataFrame(np.dot(X_wind_ons_all, weights_regr_wind_ons), columns = ['PL'])
            new_CFRs_onshore_windPL.insert(0, 'Date', wind_ons_CFR_nuts02['Date'].reset_index().drop(columns = 'index'))
            new_CFRs_onshore_windPL.to_csv(
                'new_CFRs_onshore_windPL.csv', sep=';', encoding='latin1', index=False)

        if config.ETL_RegressionCorrection_ES:

            wind_act_gen20ES = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202001010000-202101010000ES.csv',
                                                        sep=',')
            wind_act_gen21ES = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202101010000-202201010000ES.csv',
                                                        sep=',')
            wind_act_genES = wind_act_gen20ES.append(wind_act_gen21ES)
            wind_act_genES = wind_act_genES[
                ['MTU', 'Wind Offshore  - Actual Aggregated [MW]', 'Wind Onshore  - Actual Aggregated [MW]']]

            wind_act_genES['Time'] = pd.to_datetime(wind_act_genES['MTU'].str[0:10] + ' ' + wind_act_genES['MTU'].str[11:16] , format = '%d.%m.%Y %H:%M')
            wind_act_genES = wind_act_genES.drop(columns = 'MTU')
            wind_act_genES['Time'] = pd.Series(wind_act_genES['Time'].apply(lambda x: x.floor('H')))
            wind_act_genES_aggr = wind_act_genES.groupby(by=['Time']).mean().reset_index()
            wind_act_genES_aggr_max_ons = wind_act_genES_aggr['Wind Onshore  - Actual Aggregated [MW]'].max()
            wind_act_genES_aggr_max_offs = wind_act_genES_aggr['Wind Offshore  - Actual Aggregated [MW]'].max()
            wind_act_genES_aggr['Wind Onshore  - Actual Aggregated [MW]'] = wind_act_genES_aggr['Wind Onshore  - Actual Aggregated [MW]']/wind_act_genES_aggr_max_ons
            wind_act_genES_aggr['Wind Offshore  - Actual Aggregated [MW]'] = wind_act_genES_aggr[
                                                                                'Wind Offshore  - Actual Aggregated [MW]'] / wind_act_genES_aggr_max_offs

            #todo: check if we can aggregate with mean or sum
            y_wind_ons = wind_act_genES_aggr['Wind Onshore  - Actual Aggregated [MW]']
            X_wind_ons = wind_ons_CFR_nuts02_20_21[wind_ons_CFR_nuts02_20_21.columns[121:140]]
            X_wind_ons = X_wind_ons.dropna(axis = 1, how = 'all')

            X_wind_ons_all = wind_ons_CFR_nuts02[wind_ons_CFR_nuts02.columns[121:140]]
            X_wind_ons_all = X_wind_ons_all.dropna(axis=1, how='all')

            regr_wind_ons = LinearRegression(fit_intercept=False, positive=True).fit(X_wind_ons, y_wind_ons.fillna(method='ffill'))
            weights_regr_wind_ons = regr_wind_ons.coef_
            #intercept_regr_wind_ons = regr_wind_ons.intercept_

            # y_wind_offs = wind_act_genES_aggr['Wind Offshore  - Actual Aggregated [MW]']
            # X_wind_offs = wind_offs_CFR_nuts02_20_21[wind_offs_CFR_nuts02_20_21.columns[22:29]]
            # X_wind_offs = X_wind_offs.dropna(axis = 1, how = 'all')
            #
            # X_wind_offs_all = wind_offs_CFR_nuts02[wind_offs_CFR_nuts02.columns[22:29]]
            # X_wind_offs_all = X_wind_offs_all.dropna(axis = 1, how = 'all')
            #
            # regr_wind_offs = LinearRegression(fit_intercept=False,positive=True).fit(X_wind_offs, y_wind_offs)
            # weights_regr_wind_offs = regr_wind_offs.coef_
            new_CFRs_onshore_windES = pd.DataFrame(np.dot(X_wind_ons_all, weights_regr_wind_ons), columns = ['ES'])
            new_CFRs_onshore_windES.insert(0, 'Date', wind_ons_CFR_nuts02['Date'].reset_index().drop(columns = 'index'))
            # new_CFRs_offshore_wind = pd.DataFrame(np.dot(X_wind_offs_all, weights_regr_wind_offs), columns = ['ES'])
            # new_CFRs_offshore_wind.insert(0, 'Date', wind_offs_CFR_nuts02['Date'].reset_index().drop(columns = 'index'))
            new_CFRs_onshore_windES.to_csv(
                'new_CFRs_onshore_windES.csv', sep=';', encoding='latin1', index=False)
            # new_CFRs_offshore_wind.to_csv(
            #     'new_CFRs_offshore_windES.csv', sep=';', encoding='latin1', index=False)

        if config.ETL_RegressionCorrection_FR:
            # PL
            wind_act_gen20FR = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202001010000-202101010000FR.csv',
                                                        sep=',')
            wind_act_gen21FR = pd.read_csv('ENTSOEActGenWind/Actual Generation per Production Type_202101010000-202201010000FR.csv',
                                                        sep=',')
            wind_act_genFR = wind_act_gen20FR.append(wind_act_gen21FR)
            wind_act_genFR = wind_act_genFR[
                ['MTU', 'Wind Offshore  - Actual Aggregated [MW]', 'Wind Onshore  - Actual Aggregated [MW]']]

            wind_act_genFR['Time'] = pd.to_datetime(wind_act_genFR['MTU'].str[0:10] + ' ' + wind_act_genFR['MTU'].str[11:16] , format = '%d.%m.%Y %H:%M')
            wind_act_genFR = wind_act_genFR.drop(columns = 'MTU')
            wind_act_genFR['Time'] = pd.Series(wind_act_genFR['Time'].apply(lambda x: x.floor('H')))
            wind_act_genFR_aggr = wind_act_genFR.groupby(by=['Time']).mean().reset_index()
            wind_act_genFR_aggr_max_ons = wind_act_genFR_aggr['Wind Onshore  - Actual Aggregated [MW]'].max()
            wind_act_genFR_aggr['Wind Onshore  - Actual Aggregated [MW]'] = wind_act_genFR_aggr['Wind Onshore  - Actual Aggregated [MW]']/wind_act_genFR_aggr_max_ons

            #todo: check if we can aggregate with mean or sum
            y_wind_ons = wind_act_genFR_aggr['Wind Onshore  - Actual Aggregated [MW]']
            X_wind_ons = wind_ons_CFR_nuts02_20_21[wind_ons_CFR_nuts02_20_21.columns[145:167]]
            X_wind_ons = X_wind_ons.dropna(axis = 1, how = 'all')

            X_wind_ons_all = wind_ons_CFR_nuts02[wind_ons_CFR_nuts02.columns[145:167]]
            X_wind_ons_all = X_wind_ons_all.dropna(axis=1, how='all')

            regr_wind_ons = LinearRegression(fit_intercept=False, positive=True).fit(X_wind_ons, y_wind_ons.fillna(method='ffill'))
            weights_regr_wind_ons = regr_wind_ons.coef_

            new_CFRs_onshore_windFR = pd.DataFrame(np.dot(X_wind_ons_all, weights_regr_wind_ons), columns = ['FR'])
            new_CFRs_onshore_windFR.insert(0, 'Date', wind_ons_CFR_nuts02['Date'].reset_index().drop(columns = 'index'))
            new_CFRs_onshore_windFR.to_csv(
                'new_CFRs_onshore_windFR.csv', sep=';', encoding='latin1', index=False)


        new_CFRs_onshore_windNLPLES = new_CFRs_onshore_windNL.merge(new_CFRs_onshore_windPL, on = 'Date', how = 'left')
        new_CFRs_onshore_windNLPLES = new_CFRs_onshore_windNLPLES.merge(new_CFRs_onshore_windES, on = 'Date', how = 'left')
        new_CFRs_onshore_windNLPLESFR = new_CFRs_onshore_windNLPLES.merge(new_CFRs_onshore_windFR, on='Date', how='left')
        new_CFRs_onshore_windNLPLESFR.to_csv(
            'new_CFRs_onshore_windNLPLESFR.csv', sep=';', encoding='latin1', index=False)


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

        # This is current function
        #solar_pv_power_CFR_moving_avg_h2 = Preprocessor.MovingAveragesCalculatorSolarPVHourly2(solar_pv_power_CFR)

        #solar_pv_power_CFR_moving_avg_h2.to_csv(
        #     'solar_pv_power_CFR_moving_avg_h2.csv', sep=';', encoding='latin1', index=False)

        #wind_power_ons_moving_avg = Preprocessor.MovingAveragesCalculator(new_CFRs_onshore_wind)
        #wind_power_ons_moving_avg.to_csv('wind_power_ons_moving_avg.csv', sep=';', encoding='latin1', index=False)

        wind_power_ons_moving_avgFR = Preprocessor.MovingAveragesCalculator(new_CFRs_onshore_windFR)
        wind_power_ons_moving_avgFR.to_csv('wind_power_ons_moving_avgFR.csv', sep=';', encoding='latin1', index=False)

        wind_power_ons_moving_avgNLPLES = Preprocessor.MovingAveragesCalculator(new_CFRs_onshore_windNLPLES)
        wind_power_ons_moving_avgNLPLES.to_csv('wind_power_ons_moving_avgNLPLES.csv', sep=';', encoding='latin1', index=False)

        print(1)
        # wind_power_ons_moving_avg.to_csv(
        #     config.file_path_ext_ssd + 'wind_power_ons_moving_avg.csv', sep=';', encoding='latin1', index=False)

        wind_power_offs_moving_avg = Preprocessor.MovingAveragesCalculator(new_CFRs_offshore_wind)
        wind_power_offs_moving_avg.to_csv('wind_power_offs_moving_avg.csv', sep=';', encoding='latin1', index=False)
        print(1)
#        wind_power_offs_moving_avg.to_csv(
#            config.file_path_ext_ssd + 'wind_power_offs_moving_avg.csv', sep=';', encoding='latin1', index=False)

        # solar_pv_wind_power_moving_avg.to_csv(config.file_path + 'solar_pv_wind_power_moving_avg.csv')

    if config.Preprocessor_read_data_mov_avg:
        solar_pv_power_CFR_moving_avg = pd.read_csv('solar_pv_power_CFR_moving_avg_h2.csv', error_bad_lines=False,
                                                    sep=';', encoding='latin1', index_col=False)

        wind_power_ons_moving_avg = pd.read_csv('wind_power_ons_moving_avg.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)
        wind_power_ons_moving_avgNLPLES = pd.read_csv('wind_power_ons_moving_avgNLPLES.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)
        wind_power_ons_moving_avgFR = pd.read_csv('wind_power_ons_moving_avgFR.csv', error_bad_lines=False, sep=';',
                                                encoding='latin1', index_col=False)

        wind_power_offs_moving_avg = pd.read_csv('wind_power_offs_moving_avg.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)

        wind_power_ons_CFR = pd.read_csv('new_CFRs_onshore_wind.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)
        wind_power_ons_CFRNLPLESFR = pd.read_csv('new_CFRs_onshore_windNLPLESFR.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)

        wind_power_offs_CFR = pd.read_csv('new_CFRs_offshore_wind.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False)


        solar_pv_power_CFR_moving_avg['Date'] = solar_pv_power_CFR_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_ons_moving_avg['Date'] = wind_power_ons_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_ons_moving_avgNLPLES['Date'] = wind_power_ons_moving_avgNLPLES['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_ons_moving_avgFR['Date'] = wind_power_ons_moving_avgFR['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        wind_power_offs_moving_avg['Date'] = wind_power_offs_moving_avg['Date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

        wind_power_ons_moving_avg = wind_power_ons_moving_avg.merge(wind_power_ons_moving_avgNLPLES, on = 'Date', how = 'left')
        wind_power_ons_moving_avg = wind_power_ons_moving_avg.merge(wind_power_ons_moving_avgFR, on = 'Date', how = 'left')
        wind_power_ons_CFR = wind_power_ons_CFR.merge(wind_power_ons_CFRNLPLESFR, on = 'Date', how = 'left')

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


    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThresholdPVOnshoreWind(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, 'ES', [0.5, 0.5])

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThresholdPVOnshoreWind(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, 'FR', [0.5, 0.5])

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThresholdPVOnshoreWind(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, 'DE', [0.5, 0.5])

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThresholdPVOnshoreWind(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, 'NL', [0.5, 0.5])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThresholdPVOnshoreWind(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, 'PL', [0.5, 0.5])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThresholdPVOnshoreWind(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, 'ES', [0.5, 0.5])

    #
    # dunkelflaute_freq_country_i_th05ons = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
    #     installed_capacity_factor_wind_power_ons, 'DE', 0.5, 'wind_power_onshore')
    # dunkelflaute_freq_country_i_th05offs = Preprocessor.FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(
    #     installed_capacity_factor_wind_power_offs, 'DE', 0.5, 'wind_power_offshore')

    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.2, 0.2, 0.2])
    dunkelflaute_freq_country_i = Preprocessor.FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, 'DE', [0.5, 0.5, 0.5])
    print(1)

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
        i = 0
        if i == 1:
            DF_Data_all_1 = pd.read_csv(config.file_path_ext_ssd + 'DF_Data_all_t2mNL7999.csv', sep=';', encoding='latin1', header = None, error_bad_lines=False)
            DF_Data_all_2 = pd.read_csv(config.file_path_ext_ssd + 'DF_Data_all_mslNL0021.csv', sep=';', encoding='latin1', header = None)
            DF_Dates_all_1 = pd.read_csv(config.file_path_ext_ssd + 'DF_Dates_all_var_100_metre_wind_speedDE7918.csv', sep=';', encoding='latin1', header = None)
            DF_Dates_all_2 = pd.read_csv(config.file_path_ext_ssd + 'DF_Dates_all_var_100_metre_wind_speedDE1821.csv', sep=';', encoding='latin1', header = None)

            DF_Data_all_1 = DF_Data_all_1.T.values.reshape((DF_Data_all_1.shape[1], 185, 271))
            DF_Data_all_2 = DF_Data_all_2.T.values.reshape((DF_Data_all_2.shape[1], 185, 271))

            DF_Data_all = np.concatenate((DF_Data_all_1, DF_Data_all_2))
            DF_Data_all_mean = DF_Data_all.mean(axis=0)

            pd.DataFrame(DF_Data_all_mean).to_csv(
                config.file_path_ext_ssd + 'DF_Data_all_mean_' + 'msl' + 'NL' + 'corr.csv', sep=';',
                encoding='latin1', index=False,
                header=False,
                quoting=csv.QUOTE_NONE)


            print(1)
        ##data_reshaped.T.reshape((data_reshaped.shape[1], 185, 271)).mean(axis=0)



        # DF_Data_all_msl_1 = pd.read_csv(config.file_path_ext_ssd + 'DF_Data_all_msl Kopie.csv')
        # DF_Data_all_msl1521 = pd.read_csv(config.file_path_ext_ssd + 'DF_Data_all_msl1521.csv')
        # DF_Data_all_msl1921 = pd.read_csv(config.file_path_ext_ssd + 'DF_Data_all_msl1921.csv')
        #
        # DF_Data_all_msl = np.concatenate((DF_Data_all_msl_1, DF_Data_all_msl1521))
        # DF_Data_all_msl = np.concatenate((DF_Data_all_msl, DF_Data_all_msl1921))
        #
        # pd.DataFrame(DF_Data_all_msl).to_csv(
        #     config.file_path_ext_ssd + 'DF_Data_all_msl.csv', sep=';', encoding='latin1',
        #     index=False, header=False,
        #     quoting=csv.QUOTE_NONE)
        #
        # DF_Data_all_mean = DF_Data_all_msl.mean(axis=0)
        #
        # pd.DataFrame(DF_Data_all_mean).to_csv(
        #     config.file_path_ext_ssd + 'DF_Data_all_mean_' + 'msl' + '.csv', sep=';', encoding='latin1',
        #     index=False,
        #     header=False,
        #     quoting=csv.QUOTE_NONE)

        dunkelflaute_dates_DE = pd.read_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('DE') + str(
                  '0.5') + '_PVOnshoreWind_AC_dates.csv')


        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_DE, ['MSL_0000m', 'msl'], 'DE')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_DE, ['TA-_0002m', 't2m'], 'DE')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_DE, ['WS-_0010m', 'ws10'], 'DE')

        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_DE, ['WS-_0100m', 'var_100_metre_wind_speed'], 'DE')
        DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_DE, ['GHI_0000m', 'ssrd'], 'DE')
        print(1)
        # dunkelflaute_dates_FR = pd.read_csv(
        #     'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('FR') + str(
        #         '0.5') + '_PVOnshoreWind_AC_dates.csv')

        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_FR, ['MSL_0000m', 'msl'], 'FR')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_FR, ['TA-_0002m', 't2m'], 'FR')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_FR, ['WS-_0100m', 'var_100_metre_wind_speed'], 'FR')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_FR, ['WS-_0010m', 'ws10'], 'FR')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_FR, ['GHI_0000m', 'ssrd'], 'FR')

        # dunkelflaute_dates_NL = pd.read_csv(
        # 'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('NL') + str(
        #             '0.5') + '_PVOnshoreWind_AC_dates.csv')

        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_NL, ['MSL_0000m', 'msl'], 'NL')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_NL, ['TA-_0002m', 't2m'], 'NL')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_NL, ['WS-_0010m', 'ws10'], 'NL')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_NL, ['WS-_0100m', 'var_100_metre_wind_speed'], 'NL')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_NL, ['GHI_0000m', 'ssrd'], 'NL')

        dunkelflaute_dates_PL = pd.read_csv(
             'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('PL') + str(
                 '0.5') + '_PVOnshoreWind_AC_dates.csv')

        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_PL, ['MSL_0000m', 'msl'], 'PL')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_PL, ['TA-_0002m', 't2m'], 'PL')
        DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_PL, ['WS-_0100m', 'var_100_metre_wind_speed'], 'PL')
        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_PL, ['GHI_0000m', 'ssrd'], 'PL')

        #DF_Data_all_mean = DFDescriber.MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_PL, ['WS-_0010m', 'ws10'], 'PL')


        print(1)
    if config.DescriberMSLPlotter:
        import netCDF4 as nc

        fn = '/Volumes/PortableSSD/download19790102/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S197901010000_E197901312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
        ds = nc.Dataset(fn)
        #
        time = ds['time'][:]
        longitude = ds['longitude'][:]
        latitude = ds['latitude'][:]
        #
        # dunkelflaute_dates_DE = pd.read_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DE0.5_PVOnshoreWind_AC_dates.csv',
        #                                                        error_bad_lines=False, sep=';', encoding='latin1',
        #                                                        index_col=False, low_memory=False)

        # +++ DE +++
        DF_Data_all_mean_msl_DE = pd.read_csv(
            config.file_path_ext_ssd + 'DF_Data_all_mean_mslDE.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_t2m_DE = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_t2mDE.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ssrdDE = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ssrdDE.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_var_100_metre_wind_speedDE = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_var_100_metre_wind_speedDE.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ws10DE = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ws10DE.csv', header=None, index_col=None, sep=';')


        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_msl_DE, 'Germany', 'Mean sea level pressure', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_t2m_DE, 'Germany', 'Temperature (2m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ssrdDE, 'Germany', 'Solar radiation', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_var_100_metre_wind_speedDE, 'Germany', 'Wind speed (100m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ws10DE, 'Germany', 'Wind speed (10m)', longitude, latitude)

        # +++ FR +++
        DF_Data_all_mean_msl_FR = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_mslFR.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_t2m_FR = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_t2mFR.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ssrdFR = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ssrdFR.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_var_100_metre_wind_speedFR = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_var_100_metre_wind_speedFR.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ws10FR = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ws10FR.csv', header=None, index_col=None, sep=';')


        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_msl_FR, 'France', 'Mean sea level pressure', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_t2m_FR, 'France', 'Temperature (2m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ssrdFR, 'France', 'Solar radiation', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_var_100_metre_wind_speedFR, 'France', 'Wind speed (100m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ws10FR, 'France', 'Wind speed (10m)', longitude, latitude)

        # +++ NL +++
        DF_Data_all_mean_msl_NL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_mslNL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_t2m_NL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_t2mNL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ssrdNL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ssrdNL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_var_100_metre_wind_speedNL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_var_100_metre_wind_speedNL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ws10NL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ws10NL.csv', header=None, index_col=None, sep=';')


        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_msl_NL, 'Netherlands', 'Mean sea level pressure', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_t2m_NL, 'Netherlands', 'Temperature (2m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ssrdNL, 'Netherlands', 'Solar radiation', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_var_100_metre_wind_speedNL, 'Netherlands', 'Wind speed (100m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ws10NL, 'Netherlands', 'Wind speed (10m)', longitude, latitude)

        # +++ PL +++

        DF_Data_all_mean_msl_PL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_mslPL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_t2m_PL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_t2mPL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ssrdPL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ssrdPL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_var_100_metre_wind_speedPL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_var_100_metre_wind_speedPL.csv', header=None, index_col=None, sep=';')
        DF_Data_all_mean_ws10PL = pd.read_csv(
            config.file_path_ext_ssd + '/means/' + 'DF_Data_all_mean_ws10PL.csv', header=None, index_col=None, sep=';')


        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_msl_PL, 'Poland', 'Mean sea level pressure', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_t2m_PL, 'Poland', 'Temperature (2m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ssrdPL, 'Poland', 'Solar radiation', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_var_100_metre_wind_speedPL, 'Poland', 'Wind speed (100m)', longitude, latitude)
        DFDescriber.MeteoVarsPlotter(DF_Data_all_mean_ws10PL, 'Poland', 'Wind speed (10m)', longitude, latitude)


        print(1)

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