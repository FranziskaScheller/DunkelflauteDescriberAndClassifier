"""
This script contains the variables that need to be configured
"""

ETL = True
ETL_energy_vars_API = False
ETL_energy_vars_Load = True
ETL_meteo_vars = True
ETL_meteo_vars_API = False

Preprocessor = True
Preprocessor_calc_mov_avg = False

file_path_energy_vars = '/Users/franziska/PycharmProjects/DunkelflauteDescriberAndClassifier/download/'
file_names_energy_vars = ['H_ERA5_ECMW_T639_SPV_0000m_Euro_NUT0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM02.csv',
                          'H_ERA5_ECMW_T639_SPV_0000m_Euro_NUT0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM02.csv',
                          'H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
                          'H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
                          'H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
                          'H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv']

Capacity_Threshold_DF = 0.1
Min_length_DF = 24 # in hours

length_mov_avg_calc_in_days = 30

file_path = '/Users/franziska/PycharmProjects/DunkelflauteDescriberAndClassifier/'

last_year_meterology_vars = 2021
last_month_meterology_vars = 12