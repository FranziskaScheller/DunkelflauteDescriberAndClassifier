"""
This script extracts, transforms and loads the ERA5 data
"""

import config
import pandas as pd
import netCDF4 as nc
import cdsapi
from datetime import datetime, date, timedelta
import csv
import numpy as np

c = cdsapi.Client()


def EnergyVarsLoaderAPI():

    """
        This function contains code to load the energy variables
    (solar pv power generation, wind power generation off- and onshore, as capacity factor ratio and energy)
    on country and maritime country level, hourly resolution and for all years available (1979 - now)
    """

    c.retrieve(
        'sis-energy-derived-reanalysis',
        {
            'format': 'zip',
            'variable': [
                'solar_photovoltaic_power_generation', 'wind_power_generation_offshore',
                'wind_power_generation_onshore',
            ],
            'energy_product_type': [
                'capacity_factor_ratio', 'energy',
            ],
            'spatial_aggregation': [
                'country_level', 'maritime_country_level',
            ],
            'temporal_aggregation': 'hourly',
        },
        'download.zip')

def EnergyVarsLoaderFromCSV():
    """
        This function loads the energy variables data from the csv files that were returned from the API
    in the 'download.zip' file
    :return:    - solar_pv_power_CFR: dataframe with capacity factor ratios for solar and pv
                - solar_pv_power_NRG: dataframe with energy for solar and pv
                - wind_power_offs_CFR: dataframe with capacity factor ratios for offshore wind
                - wind_power_offs_NRG: dataframe with energy for offshore wind
                - wind_power_ons_CFR: dataframe with capacity factor ratios for onshore wind
                - wind_power_ons_NRG: dataframe with energy for onshore wind
    """
    # todo: change dates in file names to make it more variable
    # read csv files
    solar_pv_power_CFR = pd.read_csv(
        config.file_path_energy_vars + 'H_ERA5_ECMW_T639_SPV_0000m_Euro_NUT0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM02.csv',
        skiprows=lambda x: x in range(0, 52))
    solar_pv_power_NRG = pd.read_csv(
        config.file_path_energy_vars + 'H_ERA5_ECMW_T639_SPV_0000m_Euro_NUT0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM02.csv',
        skiprows=lambda x: x in range(0, 52))
    wind_power_offs_CFR = pd.read_csv(
        config.file_path_energy_vars + 'H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
        skiprows=lambda x: x in range(0, 52))
    wind_power_offs_NRG = pd.read_csv(
        config.file_path_energy_vars + 'H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
        skiprows=lambda x: x in range(0, 52))
    wind_power_ons_CFR = pd.read_csv(
        config.file_path_energy_vars + 'H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
        skiprows=lambda x: x in range(0, 52))
    wind_power_ons_NRG = pd.read_csv(
        config.file_path_energy_vars + 'H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
        skiprows=lambda x: x in range(0, 52))

    # wind data contains duplicates that are removed in the following
    wind_power_offs_CFR = wind_power_offs_CFR.drop_duplicates()
    wind_power_offs_NRG = wind_power_offs_NRG.drop_duplicates()
    wind_power_ons_CFR = wind_power_ons_CFR.drop_duplicates()
    wind_power_ons_NRG = wind_power_ons_NRG.drop_duplicates()

    # Transform 'Date' string in timestamp for all energy variable files
    for dataset in [solar_pv_power_CFR, solar_pv_power_NRG, wind_power_offs_CFR, wind_power_offs_NRG, wind_power_ons_CFR, wind_power_ons_NRG]:

        dataset['Date'] = dataset['Date'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

    return solar_pv_power_CFR, solar_pv_power_NRG, wind_power_offs_CFR, wind_power_offs_NRG, wind_power_ons_CFR, wind_power_ons_NRG


def FileDownloadInsights(path, file_names):

    for f in file_names:
        ds = nc.Dataset(path + f)
        print(ds)
        print(list(ds.variables.values())[3].name)

def MeterologyVarsLoaderAPI():
    """
    This function executes the API to load the meteorology variables
    - '2m_air_temperature'
    - 'pressure_at_sea_level'
    - 'surface_downwelling_shortwave_radiation'
    - 'wind_speed_at_100m'
    - 'wind_speed_at_10m'
    from the cds website.
    The zip files are stored directly in the project folder.
    """
    import cdsapi
    c = cdsapi.Client()

    years_api = [str(y) for y in range(1981, config.last_year_meterology_vars)]
    months_api = [['01', '02', ], ['03', '04', ], ['05', '06', ], ['07', '08', ], ['09', '10', ], ['11', '12', ]]

    for years in years_api:
        for months in months_api:

            c.retrieve(
                'sis-energy-derived-reanalysis',
                {
                    'format': 'zip',
                    'variable': [
                        '2m_air_temperature', 'pressure_at_sea_level', 'surface_downwelling_shortwave_radiation',
                        'wind_speed_at_100m', 'wind_speed_at_10m',
                    ],
                    'spatial_aggregation': 'original_grid',
                    'temporal_aggregation': 'hourly',
                    'year': years,
                    'month': months,
                },
                'download19810102.zip')

def MeterologyVarsLoaderAPIManually(year_start, year_end):
    import cdsapi
    c = cdsapi.Client()

    #years_api = [str(y) for y in range(1984, config.last_year_meterology_vars)]
    years_api = [str(y) for y in range(year_start, year_end)]
    months_api = [['01', '02', ], ['03', '04', ], ['05', '06', ], ['07', '08', ], ['09', '10', ], ['11', '12', ]]


    for years in years_api:
        #name = config.file_path_ext_ssd + 'download' + years + '0506.zip'
        name = 'download' + years + '0304.zip'
        c.retrieve(
                    'sis-energy-derived-reanalysis',
                    {
                        'format': 'zip',
                        'variable': [
                            '2m_air_temperature', 'pressure_at_sea_level', 'surface_downwelling_shortwave_radiation',
                            'wind_speed_at_100m', 'wind_speed_at_10m',
                        ],
                        'spatial_aggregation': 'original_grid',
                        'temporal_aggregation': 'hourly',
                        'year': years,
                        'month': ['03', '04', ],
                    },
                    name)


def MeterologyVarsLoaderGHI(years):

    for year in years:
        ind = 0
        for month in [['01', '02', '0131', '0228'], ['03', '04', '0331', '0430'], ['05', '06', '0531', '0630'], ['07', '08','0731', '0831'],
                     ['09', '10', '0930', '1031'], ['11', '12', '1130', '1231']]:

            month_start = month[0]
            month_end = month[1]
            fn = config.file_path_ext_ssd + 'download' + year + month_start + month_end + '/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S' + year + month_start + '010000_E' + year + \
                 month[2] + '2300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
            ds = nc.Dataset(fn)
            fn2 = config.file_path_ext_ssd + 'download' + year + month_start + month_end + '/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S' + year + month_end + '010000_E' + year + \
                  month[3] + '2300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
            ds2 = nc.Dataset(fn2)

            longitude = ds['longitude'][:]
            latitude = ds['latitude'][:]
            ssrd = ds['ssrd'][:].data
            ssrd2 = ds2['ssrd'][:].data

            ssrd_conc = np.concatenate((ssrd, ssrd2))
            time_conc = np.concatenate((ds['time'][:].data, ds2['time'][:].data))

            if ind == 0:
                ssrd_comp = ssrd_conc
                time_comp = time_conc

                unit = ds.variables['time'].units
                ref_date = datetime(int(unit[12:16]), int(unit[17:19]), int(unit[20:22]))
                start_date = ref_date + timedelta(hours=int(ds.variables['time'][0]))

                ind = 1
            else:
                ssrd_comp = np.concatenate((ssrd_comp, ssrd_conc))
                time_comp = np.concatenate((time_comp, time_conc))


        dates = pd.Series(start_date)
        for d in range(1, len(time_comp)):
            dates = dates.append(pd.Series(ref_date + timedelta(hours=int(time_comp[d]))))

    return ssrd_comp, dates


        # ssrd_reshaped = ssrd_comp.reshape(ssrd_comp.shape[0], -1).T.round(3)
        #
        # pd.DataFrame(ssrd_reshaped).to_csv(
        #     config.file_path_ext_ssd + 'test7.csv', sep=';', encoding='latin1', index=False, header=False,
        #     quoting=csv.QUOTE_NONE)
        # test5 = pd.read_csv(config.file_path_ext_ssd + 'test7.csv',
        #                      error_bad_lines=False, sep=';', encoding='latin1', index_col=False, header=None,
        #                      dtype='unicode', low_memory=False)

        # test4_org = test5.reshape(
        #      test5.shape[0], test5.shape[1] // ssrd_comp.shape[2], ssrd_comp.shape[2])
        #
        # pd.DataFrame(dates).to_csv(
        #     config.file_path_ext_ssd + 'dates.csv', sep=';', encoding='latin1', index=False, header=False,
        #     quoting=csv.QUOTE_NONE)
        # dates_test = pd.read_csv(config.file_path_ext_ssd + 'dates.csv',
        #                      error_bad_lines=False, sep=';', encoding='latin1', index_col=False, header=None,
        #                      dtype='unicode', low_memory=False)


def MeterologyVarsLoader(years, var):

    for year in years:

        fn = config.file_path_ext_ssd + 'download' + year + '0102/H_ERA5_ECMW_T639_' + var[0] + '_Euro_025d_S' + year + '01010000_E' + year + '12312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
        ds = nc.Dataset(fn)

        data_var = ds[var[1]][:].data


        unit = ds.variables['time'].units
        ref_date = datetime(int(unit[12:16]), int(unit[17:19]), int(unit[20:22]))
        start_date = ref_date + timedelta(hours=int(ds.variables['time'][0]))

        dates = pd.Series(start_date)
        for d in range(1, len(ds['time'][:].data)):
            dates = dates.append(pd.Series(ref_date + timedelta(hours=int(ds['time'][:].data[d]))))

    return data_var, dates

            # data_reshaped = data.reshape(data.shape[0], -1).T.round(3)
            #
            # pd.DataFrame(data_reshaped).to_csv(
            #     config.file_path_ext_ssd + var[0] + year + '.csv', sep=';', encoding='latin1', index=False, header=False,
            #     quoting=csv.QUOTE_NONE)
            # test5 = pd.read_csv(config.file_path_ext_ssd + var[0] + year + '.csv',
            #                      error_bad_lines=False, sep=';', encoding='latin1', index_col=False, header=None,
            #                      dtype='unicode', low_memory=False)
            # print(11)
            # test4_org = test5.reshape(
            #      test5.shape[0], test5.shape[1] // ssrd_comp.shape[2], ssrd_comp.shape[2])
            #
            # pd.DataFrame(dates).to_csv(
            #     config.file_path_ext_ssd + 'dates.csv', sep=';', encoding='latin1', index=False, header=False,
            #     quoting=csv.QUOTE_NONE)
            # dates_test = pd.read_csv(config.file_path_ext_ssd + 'dates.csv',
            #                      error_bad_lines=False, sep=';', encoding='latin1', index_col=False, header=None,
            #                      dtype='unicode', low_memory=False)

def MeterologyVarsReaderGHI(years):

    for year in years:
        ind = 0
        for month in [['01', '02', '0131', '0228'], ['03', '04', '0331', '0430'], ['05', '06', '0531', '0630'], ['07', '08','0731', '0831'],
                     ['09', '10', '0930', '1031'], ['11', '12', '1130', '1231']]:

            month_start = month[0]
            month_end = month[1]
            fn = config.file_path_ext_ssd + 'download' + year + month_start + month_end + '/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S' + year + month_start + '010000_E' + year + \
                 month[2] + '2300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
            ds = nc.Dataset(fn)
            fn2 = config.file_path_ext_ssd + 'download' + year + month_start + month_end + '/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S' + year + month_end + '010000_E' + year + \
                  month[3] + '2300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
            ds2 = nc.Dataset(fn2)

            ssrd = ds['ssrd'][:].data
            ssrd2 = ds2['ssrd'][:].data

            ssrd_conc = np.concatenate((ssrd, ssrd2))
            time_conc = np.concatenate((ds['time'][:].data, ds2['time'][:].data))

            if ind == 0:
                ssrd_comp = ssrd_conc
                time_comp = time_conc

                unit = ds.variables['time'].units
                ref_date = datetime(int(unit[12:16]), int(unit[17:19]), int(unit[20:22]))
                start_date = ref_date + timedelta(hours=int(ds.variables['time'][0]))

                ind = 1
            else:
                ssrd_comp = np.concatenate((ssrd_comp, ssrd_conc))
                time_comp = np.concatenate((time_comp, time_conc))

        dates = pd.Series(start_date)
        for d in range(1, len(time_comp)):
            dates = dates.append(pd.Series(ref_date + timedelta(hours=int(time_comp[d]))))

    return ssrd_comp, dates


def MeterologyVarsReader(years, var):
    ind2 = 0
    for year in years:

        ind = 0
        if ((year in [2019, 2020, 2021, 2021, 2022]) or (var[0] == 'GHI_0000m')):

            print(1)
            for month in [['01', '02', '0131', '0228'], ['03', '04', '0331', '0430'], ['05', '06', '0531', '0630'],
                              ['07', '08', '0731', '0831'],
                              ['09', '10', '0930', '1031'], ['11', '12', '1130', '1231']]:

                month_start = month[0]
                month_end = month[1]
                fn = config.file_path_ext_ssd + 'download' + str(year) + month_start + month_end + '/H_ERA5_ECMW_T639_' + var[0] + '_Euro_025d_S' + str(year) + month_start + '010000_E' + str(year) + \
                        month[2] + '2300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
                ds = nc.Dataset(fn)
                if (((year == 2020) or (year == 1980) or (year == 1984) or (year == 1988) or (year == 1992) or (year == 1996) or (year == 2000) or (year == 2004) or (year == 2008) or (year == 2012) or (year == 2016)) & (month_end == '02')) :
                    fn2 = config.file_path_ext_ssd + 'download' + str(year) + month_start + month_end + '/H_ERA5_ECMW_T639_' + var[0] + '_Euro_025d_S' + str(year) + month_end + '010000_E' + str(year) + \
                            '0229' + '2300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
                else:
                    fn2 = config.file_path_ext_ssd + 'download' + str(year) + month_start + month_end + '/H_ERA5_ECMW_T639_' + var[0] + '_Euro_025d_S' + str(year) + month_end + '010000_E' + str(year) + \
                            month[3] + '2300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
                ds2 = nc.Dataset(fn2)

                if ((year >= 2018) & (var[1] == 'ws10')):
                    ssrd = ds['var_10_metre_wind_speed'][:].data
                    ssrd2 = ds2['var_10_metre_wind_speed'][:].data
                else:
                    ssrd = ds[var[1]][:].data
                    ssrd2 = ds2[var[1]][:].data

                ssrd_conc = np.concatenate((ssrd, ssrd2))
                time_conc = np.concatenate((ds['time'][:].data, ds2['time'][:].data))

                if ind == 0:
                    ssrd_comp = ssrd_conc
                    time_comp = time_conc

                    unit = ds.variables['time'].units
                    ref_date = datetime(int(unit[12:16]), int(unit[17:19]), int(unit[20:22]))
                    start_date = ref_date + timedelta(hours=int(ds.variables['time'][0]))

                    ind = 1
                else:
                    ssrd_comp = np.concatenate((ssrd_comp, ssrd_conc))
                    time_comp = np.concatenate((time_comp, time_conc))

            dates = pd.Series(start_date)
            for d in range(1, len(time_comp)):
                dates = dates.append(pd.Series(ref_date + timedelta(hours=int(time_comp[d]))))

            dates_all = dates
            data_var_all = ssrd_comp
        else:

            fn = config.file_path_ext_ssd + 'download' + str(year) + '0102/H_ERA5_ECMW_T639_' + str(var[0]) + '_Euro_025d_S' + str(year) + '01010000_E' + str(year) + '12312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
            ds = nc.Dataset(fn)
            if ((year >= 2019) & (var[1] == 'ws10')):
                data_var = ds['var_10_metre_wind_speed'][:].data
            else:
                data_var = ds[var[1]][:].data

            unit = ds.variables['time'].units
            ref_date = datetime(int(unit[12:16]), int(unit[17:19]), int(unit[20:22]))
            start_date = ref_date + timedelta(hours=int(ds.variables['time'][0]))

            dates = pd.Series(start_date)
            for d in range(1, len(ds['time'][:].data)):
                dates = dates.append(pd.Series(ref_date + timedelta(hours=int(ds['time'][:].data[d]))))

            if ind2 == 0:
                data_var_all = data_var
                dates_all = dates
                ind2 = 1
            else:
                data_var_all = np.concatenate((data_var_all, data_var))
                dates_all = np.concatenate((dates_all, dates))

    return data_var_all, dates_all



