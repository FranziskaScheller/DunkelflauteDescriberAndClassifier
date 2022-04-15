"""
This script extracts, transforms and loads the ERA5 data
"""

import config
import pandas as pd
import netCDF4 as nc
import cdsapi
from datetime import datetime, date
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

def MeterologyVarsLoaderAPI2():
    import cdsapi
    c = cdsapi.Client()

    years_api = [str(y) for y in range(1984, config.last_year_meterology_vars)]
    months_api = [['01', '02', ], ['03', '04', ], ['05', '06', ], ['07', '08', ], ['09', '10', ], ['11', '12', ]]


    for years in years_api:
        name = 'download' + years + '0102.zip'
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
                        'month': ['01', '02', ],
                    },
                    name)


    # import cdsapi
    #
    # c = cdsapi.Client()
    #
    # c.retrieve(
    #     'sis-energy-derived-reanalysis',
    #     {
    #         'format': 'zip',
    #         'variable': [
    #             '2m_air_temperature', 'pressure_at_sea_level', 'surface_downwelling_shortwave_radiation',
    #             'wind_speed_at_100m', 'wind_speed_at_10m',
    #         ],
    #         'spatial_aggregation': 'original_grid',
    #         'temporal_aggregation': 'hourly',
    #         'year': '1979',
    #         'month': [
    #             '01', '02',
    #         ],
    #     },
    #     'download.zip')
#
#
# year_today = date.today().year()
# month_today = date.today().month()


