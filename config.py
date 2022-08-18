"""
+++ This script contains the variables that need to be configured +++
"""

"""
Select the modules that you want to execute in this run 
"""
# If true, the ETL is executed
ETL = True
# If true, the individual components of the ETL is executed
ETL_energy_vars_API = False
ETL_energy_vars_Load = True
ETL_meteo_vars = False
ETL_meteo_vars_API = False
ETL_meteo_vars_Write = False
ETL_meteo_vars_Reader = False
ETL_RegressionCorrection = False
ETL_RegressionCorrection_DE = True
ETL_RegressionCorrection_NL = True
ETL_RegressionCorrection_PL = True
ETL_RegressionCorrection_ES = True
ETL_RegressionCorrection_FR = True

# If true, the Preprocessor is executed
Preprocessor = False
Preprocessor_calc_mov_avg = False
Preprocessor_read_data_mov_avg = True
Preprocessor_installed_capacity_corrector = True
calc_summary_measures = False
# If true, the Dunkelflaute Describer is executed
Describer = True
# If true, the individual components of the Dunkelflaute Describer is executed

DescriberMSLDataCalculator = False
DescriberMSLPlotter = True
DescriberTAPlotter = True
DescriberWS10Plotter = True
DescriberWS100Plotter = True
DescriberGHIPlotter = True

"""
Specify file locations and file names (which should normally not change)
"""
file_path_energy_vars = '/Users/franziska/PycharmProjects/DunkelflauteDescriberAndClassifier/download/'
file_names_energy_vars = ['H_ERA5_ECMW_T639_SPV_0000m_Euro_NUT0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM02.csv',
                          'H_ERA5_ECMW_T639_SPV_0000m_Euro_NUT0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM02.csv',
                          'H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
                          'H_ERA5_ECMW_T639_WOF_0100m_Euro_MAR0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
                          'H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202203312300_CFR_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv',
                          'H_ERA5_ECMW_T639_WON_0100m_Euro_NUT0_S197901010000_E202203312300_NRG_TIM_01h_NA-_noc_org_NA_NA---_NA---_PhM01.csv']

file_path_ext_ssd = '/Volumes/PortableSSD/'
file_path = '/Users/franziska/PycharmProjects/DunkelflauteDescriberAndClassifier/'

"""
Dunkelflaute specific arguments 
"""
Capacity_Threshold_DF = 0.2
Min_length_DF = 24 # in hours
length_mov_avg_calc_in_days = 30

#range_lengths_DF_hist = range(1, 72)
range_lengths_DF_hist = range(24, 120)
"""
Optional parameters
"""

last_year_meterology_vars = 2021
last_month_meterology_vars = 12