from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import config
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

### Create MasterTable ###
msl_aggr_DE_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland.csv', sep=';')
msl_aggr_DE_79to21['Dates'] = msl_aggr_DE_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_DE_79to21 = msl_aggr_DE_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp_mean', 'std': 'mslp_std'})

msl_aggr_GWL_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_msl_all.csv', sep=';')
msl_aggr_GWL_79to21['Date'] = msl_aggr_GWL_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

msl_aggr_NL_79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslNederland79to90.csv', sep = ';', encoding='latin1')
msl_aggr_NL_91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslNederland91to00.csv', sep = ';', encoding='latin1')
msl_aggr_NL_01to10 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslNederland01to10.csv', sep = ';', encoding='latin1')
msl_aggr_NL_11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslNederland11to21.csv', sep = ';', encoding='latin1')

msl_aggr_NL_79to21 = pd.concat([msl_aggr_NL_79to90, msl_aggr_NL_91to00, msl_aggr_NL_01to10, msl_aggr_NL_11to21])

msl_aggr_NL_79to21['Dates'] = msl_aggr_NL_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_NL_79to21 = msl_aggr_NL_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp_mean_NL', 'std': 'mslp_std_NL'})

msl_aggr_PL_79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska79to90.csv', sep = ';', encoding='latin1')
msl_aggr_PL_91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska91to00.csv', sep = ';', encoding='latin1')
msl_aggr_PL_01to10 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska01to10.csv', sep = ';', encoding='latin1')
msl_aggr_PL_11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska11to21.csv', sep = ';', encoding='latin1')

msl_aggr_PL_79to21 = pd.concat([msl_aggr_PL_79to90, msl_aggr_PL_91to00, msl_aggr_PL_01to10, msl_aggr_PL_11to21])

msl_aggr_PL_79to21['Dates'] = msl_aggr_PL_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_PL_79to21 = msl_aggr_PL_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp_mean_PL', 'std': 'mslp_std_PL'})

msl_aggr_FR_79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance79to90.csv', sep = ';', encoding='latin1')
msl_aggr_FR_91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance91to00.csv', sep = ';', encoding='latin1')
msl_aggr_FR_01to10 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance01to10.csv', sep = ';', encoding='latin1')
msl_aggr_FR_11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance11to21.csv', sep = ';', encoding='latin1')

msl_aggr_FR_79to21 = pd.concat([msl_aggr_FR_79to90, msl_aggr_FR_91to00, msl_aggr_FR_01to10, msl_aggr_FR_11to21])

msl_aggr_FR_79to21['Dates'] = msl_aggr_FR_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_FR_79to21 = msl_aggr_FR_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp_mean_FR', 'std': 'mslp_std_FR'})

t2m_aggr_DE_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_t2mDeutschland.csv', sep=';')
t2m_aggr_DE_79to21['Dates'] = t2m_aggr_DE_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
t2m_aggr_DE_79to21 = t2m_aggr_DE_79to21.rename(columns = {'Dates': 'Date', 'mean': 't2m_mean', 'std': 't2m_std'})

# ---- Installed CF
installed_capacity_factor_solar_pv_power = pd.read_csv('installed_capacity_factor_solar_pv_power_h2.csv',
                                                       error_bad_lines=False, sep=';', encoding='latin1',
                                                       index_col=False, low_memory=False)
installed_capacity_factor_wind_power_ons = pd.read_csv('installed_capacity_factor_wind_power_ons.csv',
                                                       error_bad_lines=False, sep=';', encoding='latin1',
                                                       index_col=False, low_memory=False)

installed_capacity_factor_solar_pv_power['Date'] = installed_capacity_factor_solar_pv_power['Date'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
installed_capacity_factor_wind_power_ons['Date'] = installed_capacity_factor_wind_power_ons['Date'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power.fillna(0)
installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power.round(3)
installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons.round(3)

#installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power[installed_capacity_factor_solar_pv_power['Date'].dt.year.isin(range(2007, 2022))].reset_index().drop(columns = 'index')
installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power[['Date', 'DE']]

#installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons[installed_capacity_factor_wind_power_ons['Date'].dt.year.isin(range(2007, 2022))].reset_index().drop(columns = 'index')
installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons[['Date', 'DE']]


# dunkelflaute_dates_DE = pd.read_csv(
#     'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('DE') + str(
#         '0.5') + '_PVOnshoreWind_AC_dates.csv')
#
# dunkelflaute_dates_DE['DFDates'] = dunkelflaute_dates_DE['0'].apply(
#     lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
# DF_indices = msl_aggr_DE_79to21['Date'].isin(dunkelflaute_dates_DE['DFDates'])
# DF_indices_0_1_encoding = DF_indices.apply(lambda x: int(x))


mastertableDFclassifier = msl_aggr_DE_79to21.merge(t2m_aggr_DE_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_GWL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_PL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_NL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_FR_79to21, on = 'Date', how = 'left')

mastertableDFclassifier = mastertableDFclassifier.merge(installed_capacity_factor_wind_power_ons, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.rename(columns = {'DE': 'AdjCF_OnsWind'})
mastertableDFclassifier = mastertableDFclassifier.merge(installed_capacity_factor_solar_pv_power, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.rename(columns = {'DE': 'AdjCF_Solar'})

#mastertableDFclassifier['DF_Indicator_h'] = DF_indices_0_1_encoding
mastertableDFclassifier['Flaute_Indicator'] = (mastertableDFclassifier['AdjCF_OnsWind'] <= 0.25)
mastertableDFclassifier['Dunkel_Indicator'] = (mastertableDFclassifier['AdjCF_Solar'] <= 0.25)

mastertableDFclassifier['Flaute_Indicator'] = mastertableDFclassifier['Flaute_Indicator'].apply(lambda x: int(x))
mastertableDFclassifier['Dunkel_Indicator'] = mastertableDFclassifier['Dunkel_Indicator'].apply(lambda x: int(x))

mastertableDFclassifier['Month'] = mastertableDFclassifier['Date'].apply(lambda x: x.month)
mastertableDFclassifier['Hour'] = mastertableDFclassifier['Date'].apply(lambda x: x.hour)
ohe = OneHotEncoder().fit_transform(X = pd.DataFrame(mastertableDFclassifier['Date'].apply(lambda x: x.month).values)).toarray()
mastertableDFclassifier[['Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec']] = ohe
# --- Add Lags ---
daily_mean = mastertableDFclassifier.set_index('Date')[['mslp_mean', 'mslp_std']].groupby(pd.Grouper(freq='d')).mean().rename({'mslp_mean': 'mslp_daily_mean', 'mslp_std': 'mslp_daily_std' }, axis = 'columns').reset_index()
mastertableDFclassifier = mastertableDFclassifier.merge(daily_mean, on = 'Date', how = 'left')

daily_max = mastertableDFclassifier.set_index('Date')[['mslp_mean', 'mslp_std']].groupby(pd.Grouper(freq='d')).max().rename({'mslp_mean': 'mslp_daily_mean_max', 'mslp_std': 'mslp_daily_std_max' }, axis = 'columns').reset_index()
mastertableDFclassifier = mastertableDFclassifier.merge(daily_max, on = 'Date', how = 'left')

daily_min = mastertableDFclassifier.set_index('Date')[['mslp_mean', 'mslp_std']].groupby(pd.Grouper(freq='d')).min().rename({'mslp_mean': 'mslp_daily_mean_min', 'mslp_std': 'mslp_daily_std_min' }, axis = 'columns').reset_index()
mastertableDFclassifier = mastertableDFclassifier.merge(daily_min, on = 'Date', how = 'left')

mastertableDFclassifier[['mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min']] = mastertableDFclassifier[['mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min']].fillna(method="ffill")

mastertableDFclassifier['Lag_1h_mslp_mean'] = mastertableDFclassifier['mslp_mean'].shift(1)
mastertableDFclassifier['Lag_2h_mslp_mean'] = mastertableDFclassifier['mslp_mean'].shift(2)
mastertableDFclassifier['Lag_5h_mslp_mean'] = mastertableDFclassifier['mslp_mean'].shift(5)
mastertableDFclassifier['Lag_24h_mslp_mean'] = mastertableDFclassifier['mslp_mean'].shift(24)
mastertableDFclassifier['Lag_48h_mslp_mean'] = mastertableDFclassifier['mslp_mean'].shift(48)
mastertableDFclassifier['Lag_120h_mslp_mean'] = mastertableDFclassifier['mslp_mean'].shift(120)

mastertableDFclassifier['Lag_24h_mslp_mean_PL'] = mastertableDFclassifier['mslp_mean_PL'].shift(24)
mastertableDFclassifier['Lag_24h_mslp_mean_NL'] = mastertableDFclassifier['mslp_mean_NL'].shift(24)
mastertableDFclassifier['Lag_24h_mslp_mean_FR'] = mastertableDFclassifier['mslp_mean_FR'].shift(24)

mastertableDFclassifier['Lag_120h_mslp_mean_PL'] = mastertableDFclassifier['mslp_mean_PL'].shift(120)
mastertableDFclassifier['Lag_120h_mslp_mean_NL'] = mastertableDFclassifier['mslp_mean_NL'].shift(120)
mastertableDFclassifier['Lag_120h_mslp_mean_FR'] = mastertableDFclassifier['mslp_mean_FR'].shift(120)

mastertableDFclassifier['Lag_1h_mslp_std'] = mastertableDFclassifier['mslp_std'].shift(1)
mastertableDFclassifier['Lag_2h_mslp_std'] = mastertableDFclassifier['mslp_std'].shift(2)
mastertableDFclassifier['Lag_5h_mslp_std'] = mastertableDFclassifier['mslp_std'].shift(5)
mastertableDFclassifier['Lag_24h_mslp_std'] = mastertableDFclassifier['mslp_std'].shift(24)
mastertableDFclassifier['Lag_48h_mslp_std'] = mastertableDFclassifier['mslp_std'].shift(48)
mastertableDFclassifier['Lag_120h_mslp_std'] = mastertableDFclassifier['mslp_std'].shift(120)

mastertableDFclassifier['Lag_120h_mslp_std_PL'] = mastertableDFclassifier['mslp_std_PL'].shift(120)
mastertableDFclassifier['Lag_120h_mslp_std_NL'] = mastertableDFclassifier['mslp_std_NL'].shift(120)
mastertableDFclassifier['Lag_120h_mslp_std_FR'] = mastertableDFclassifier['mslp_std_FR'].shift(120)

mastertableDFclassifier['Lag_24h_t2m_mean'] = mastertableDFclassifier['t2m_mean'].shift(24)
mastertableDFclassifier['Lag_48h_t2m_mean'] = mastertableDFclassifier['t2m_mean'].shift(48)
mastertableDFclassifier['Lag_120h_t2m_mean'] = mastertableDFclassifier['t2m_mean'].shift(120)

mastertableDFclassifier['Lag_24h_t2m_std'] = mastertableDFclassifier['t2m_std'].shift(24)
mastertableDFclassifier['Lag_48h_t2m_std'] = mastertableDFclassifier['t2m_std'].shift(48)
mastertableDFclassifier['Lag_120h_t2m_std'] = mastertableDFclassifier['t2m_std'].shift(120)

for col in msl_aggr_GWL_79to21.columns[1:-1]:
    mastertableDFclassifier[str(col) + '_Lag_24h'] = mastertableDFclassifier[str(col)].shift(24)
    mastertableDFclassifier[str(col) + '_Lag_120h'] = mastertableDFclassifier[str(col)].shift(120)


#mastertableDFclassifier = mastertableDFclassifier[mastertableDFclassifier['Date'].apply(lambda x: x.hour) == 11]
# Test
# mastertableDFclassifier = mastertableDFclassifier.set_index('Date').groupby(pd.Grouper(freq='d')).mean()
# mastertableDFclassifier = mastertableDFclassifier.reset_index()
# mastertableDFclassifier = mastertableDFclassifier.dropna()
# mastertableDFclassifier = mastertableDFclassifier.reset_index().drop(columns = 'index')
# mastertableDFclassifier['DF_Indicator'] = 0
# mastertableDFclassifier['DF_Indicator'][mastertableDFclassifier['DF_Indicator_h'] >= 0.5] = 1

mastertableDFclassifier = mastertableDFclassifier.dropna()
mastertableDFclassifier = mastertableDFclassifier.reset_index().drop(columns = 'index')

#X = mastertableDFclassifier[['mslp_mean', 'mslp_std', 't2m_mean', 'Month', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std', 'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR']]
#X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Month','Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL']]
#X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec', 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std', 'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR', 'Lag_24h_mslp_mean_PL', 'Lag_24h_mslp_mean_NL',  'Lag_24h_mslp_mean_FR']]
#X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec', 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std',
#                             'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR', 'Lag_24h_mslp_mean_PL', 'Lag_24h_mslp_mean_NL',  'Lag_24h_mslp_mean_FR', 'mean_Greenland' ,'std_Greenland', 'mean_Iceland' ,'std_Iceland', 'mean_British_Isles', 'std_British_Isles', 'mean_Mediterranean_Sea', 'std_Mediterranean_Sea' ,'mean_Sea_west_Iberian_Peninsula', 'std_Sea_west_Iberian_Peninsula' ,'mean_Norwegian_Sea' ,'std_Norwegian_Sea', 'mean_North_Sea', 'std_North_Sea', 'mean_Western_Russia', 'std_Western_Russia', 'mean_Sweden', 'std_Sweden', 'mean_Tyrrhenian_Sea', 'std_Tyrrhenian_Sea']]
# So far best below!
# X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec',
#                                 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std',
#                                 'mean_Greenland' ,'std_Greenland', 'mean_Iceland' ,'std_Iceland', 'mean_British_Isles', 'std_British_Isles', 'mean_Mediterranean_Sea', 'std_Mediterranean_Sea' ,'mean_Sea_west_Iberian_Peninsula', 'std_Sea_west_Iberian_Peninsula' ,'mean_Norwegian_Sea' ,'std_Norwegian_Sea', 'mean_North_Sea', 'std_North_Sea', 'mean_Western_Russia', 'std_Western_Russia', 'mean_Sweden', 'std_Sweden', 'mean_Tyrrhenian_Sea', 'std_Tyrrhenian_Sea']]

# X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec',
#                                 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std',
#                                 'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR', 'Lag_24h_mslp_mean_PL', 'Lag_24h_mslp_mean_NL',  'Lag_24h_mslp_mean_FR',
#                                 'mean_Greenland' , 'mean_Iceland' , 'mean_British_Isles',  'mean_Mediterranean_Sea', 'mean_Sea_west_Iberian_Peninsula','mean_Norwegian_Sea' , 'mean_North_Sea', 'mean_Western_Russia',  'mean_Sweden', 'mean_Tyrrhenian_Sea']]

#X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec', 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std', 'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR', 'Lag_24h_mslp_mean_PL', 'Lag_24h_mslp_mean_NL',  'Lag_24h_mslp_mean_FR']]

# X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec', 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std', 'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR', 'Lag_24h_mslp_mean_PL', 'Lag_24h_mslp_mean_NL',  'Lag_24h_mslp_mean_FR'
#                              , 'mean_Greenland' , 'mean_British_Isles',  'mean_Mediterranean_Sea', 'mean_Sea_west_Iberian_Peninsula','mean_Norwegian_Sea' , 'mean_North_Sea', 'mean_Western_Russia',  'mean_Sweden'
#                              , 'mean_Greenland_Lag_24h' , 'mean_British_Isles_Lag_24h',  'mean_Mediterranean_Sea_Lag_24h', 'mean_Sea_west_Iberian_Peninsula_Lag_24h','mean_Norwegian_Sea_Lag_24h' , 'mean_North_Sea_Lag_24h', 'mean_Western_Russia_Lag_24h',  'mean_Sweden_Lag_24h'
#                             , 'mean_Greenland_Lag_120h', 'mean_British_Isles_Lag_120h', 'mean_Mediterranean_Sea_Lag_120h', 'mean_Sea_west_Iberian_Peninsula_Lag_120h', 'mean_Norwegian_Sea_Lag_120h', 'mean_North_Sea_Lag_120h', 'mean_Western_Russia_Lag_120h', 'mean_Sweden_Lag_120h']]

X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Ind_Jan', 'Ind_Feb', 'Ind_Mar', 'Ind_Apr', 'Ind_May', 'Ind_Jun', 'Ind_Jul', 'Ind_Aug', 'Ind_Sep', 'Ind_Oct', 'Ind_Nov', 'Ind_Dec', 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std', 'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR', 'Lag_24h_mslp_mean_PL', 'Lag_24h_mslp_mean_NL',  'Lag_24h_mslp_mean_FR'
                             , 'mean_Greenland' , 'mean_British_Isles',  'mean_Mediterranean_Sea', 'mean_Sea_west_Iberian_Peninsula','mean_Norwegian_Sea' , 'mean_North_Sea', 'mean_Western_Russia',  'mean_Sweden'
                            , 'mean_Greenland_Lag_24h' , 'mean_British_Isles_Lag_24h',  'mean_Mediterranean_Sea_Lag_24h', 'mean_Sea_west_Iberian_Peninsula_Lag_24h','mean_Norwegian_Sea_Lag_24h' , 'mean_North_Sea_Lag_24h', 'mean_Western_Russia_Lag_24h',  'mean_Sweden_Lag_24h']]

y = mastertableDFclassifier['Flaute_Indicator']

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle = False)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

print(1)
# neigh_50 = KNeighborsClassifier(n_neighbors=50)
# knn_fit_50 = neigh_50.fit(X_train, y_train)
# knn_pred_50 = knn_fit_50.predict_proba(X_test)
#
# knn_pred_m_50 = pd.DataFrame(knn_pred_50)
# knn_pred_m_50['real'] = y_test.values
#
# knn_res_y_50 = np.zeros(len(y_test))
# knn_res_y_50[knn_pred_50[:,1] >= 0.005] = 1
# knn_res_y_50 = pd.DataFrame(knn_res_y_50)
# #
# conf_mat_50 = confusion_matrix(y_test, knn_res_y_50.values)
# precision_50 = precision_score(y_test, knn_res_y_50.values)
# recall_50 = recall_score(y_test, knn_res_y_50.values)
#
# neigh_70 = KNeighborsClassifier(n_neighbors=70)
# knn_fit_70 = neigh_70.fit(X_train, y_train)
# knn_pred_70 = knn_fit_70.predict_proba(X_test)
#
# knn_pred_m_70 = pd.DataFrame(knn_pred_70)
# knn_pred_m_70['real'] = y_test.values
#
# knn_res_y_70 = np.zeros(len(y_test))
# knn_res_y_70[knn_pred_70[:,1] >= 0.05] = 1
# knn_res_y_70 = pd.DataFrame(knn_res_y_70)
#
# conf_mat_70 = confusion_matrix(y_test, knn_res_y_70.values)
# precision_70 = precision_score(y_test, knn_res_y_70.values)
# recall_70 = recall_score(y_test, knn_res_y_70.values)

# neigh_30 = KNeighborsClassifier(n_neighbors=30)
# knn_fit_30 = neigh_30.fit(X_train, y_train)
# knn_pred_30 = knn_fit_30.predict_proba(X_test)
#
# knn_pred_m_30 = pd.DataFrame(knn_pred_30)
# knn_pred_m_30['real'] = y_test.values
#
# knn_res_y_30 = np.zeros(len(y_test))
# knn_res_y_30[knn_pred_30[:,1] >= 0.05] = 1
# knn_res_y_30 = pd.DataFrame(knn_res_y_30)
#
# conf_mat_30 = confusion_matrix(y_test, knn_res_y_30.values)
# precision_30 = precision_score(y_test, knn_res_y_30.values)
# recall_30 = recall_score(y_test, knn_res_y_30.values)

# neigh_20 = KNeighborsClassifier(n_neighbors=20)
# knn_fit_20 = neigh_20.fit(X_train, y_train)
# knn_pred_20 = knn_fit_20.predict_proba(X_test)
#
# knn_pred_m_20 = pd.DataFrame(knn_pred_20)
# knn_pred_m_20['real'] = y_test.values
#
# knn_res_y_20 = np.zeros(len(y_test))
# knn_res_y_20[knn_pred_20[:,1] >= 0.01] = 1
# knn_res_y_20 = pd.DataFrame(knn_res_y_20)
#
# conf_mat_20 = confusion_matrix(y_test, knn_res_y_20.values)
# precision_20 = precision_score(y_test, knn_res_y_20.values)
# recall_20 = recall_score(y_test, knn_res_y_20.values)
#
# neigh_10 = KNeighborsClassifier(n_neighbors=10)
# knn_fit_10 = neigh_10.fit(X_train, y_train)
# knn_pred_10 = knn_fit_10.predict_proba(X_test)
#
# knn_pred_m_10 = pd.DataFrame(knn_pred_10)
# knn_pred_m_10['real'] = y_test.values
#
# knn_res_y_10 = np.zeros(len(y_test))
# knn_res_y_10[knn_pred_10[:,1] >= 0.01] = 1
# knn_res_y_10 = pd.DataFrame(knn_res_y_10)
#
# conf_mat_10 = confusion_matrix(y_test, knn_res_y_10.values)
# precision_10 = precision_score(y_test, knn_res_y_10.values)
# recall_10 = recall_score(y_test, knn_res_y_10.values)

# neigh_5 = KNeighborsClassifier(n_neighbors=5)
# knn_fit_5 = neigh_5.fit(X_train, y_train)
# knn_pred_5 = knn_fit_5.predict_proba(X_test)
#
# knn_pred_m_5 = pd.DataFrame(knn_pred_5)
# knn_pred_m_5['real'] = y_test.values
#
# knn_res_y_5 = np.zeros(len(y_test))
# knn_res_y_5[knn_pred_5[:,1] >= 0.005] = 1
# knn_res_y_5 = pd.DataFrame(knn_res_y_5)
#
# conf_mat_5 = confusion_matrix(y_test, knn_res_y_5.values)
# precision_5 = precision_score(y_test, knn_res_y_5.values)
# recall_5 = recall_score(y_test, knn_res_y_5.values)

rfc = RandomForestClassifier(random_state=0, min_samples_leaf = 2, n_estimators= 200).fit(X_train, y_train)
pred_rfc = rfc.predict_proba(X_test)
#
clf_pred_m_rfc = pd.DataFrame(pred_rfc)
clf_pred_m_rfc['real'] = y_test.values
#
clf_res_y_rfc = np.zeros(len(y_test))
clf_res_y_rfc[pred_rfc[:,1] >= 0.006] = 1
clf_res_y_rfc = pd.DataFrame(clf_res_y_rfc)
#
conf_mat_logreg_rfc = confusion_matrix(y_test, clf_res_y_rfc.values)
precision_logreg_rfc = precision_score(y_test, clf_res_y_rfc.values)
recall_logreg_rfc = recall_score(y_test, clf_res_y_rfc.values)


clf = LogisticRegression(random_state=0, max_iter=600).fit(X_train, y_train)
pred_cfl = clf.predict_proba(X_test)
clf_pred_m = pd.DataFrame(pred_cfl)
clf_pred_m['real'] = y_test.values

clf_res_y = np.zeros(len(y_test))
clf_res_y[pred_cfl[:,1] >= 0.006] = 1
clf_res_y = pd.DataFrame(clf_res_y)

conf_mat_logreg = confusion_matrix(y_test, clf_res_y.values)
precision_logreg = precision_score(y_test, clf_res_y.values)
recall_logreg = recall_score(y_test, clf_res_y.values)

#---
res_eval_df_probs = pd.DataFrame(mastertableDFclassifier[mastertableDFclassifier.index.isin(X_test.index)].Date, columns = ['Date'])
res_eval_df_probs['DF_ind'] = y_test
res_eval_df_probs['DF_pred'] = pred_cfl[:,1]

res_eval_df_probs_rfc = pd.DataFrame(mastertableDFclassifier[mastertableDFclassifier.index.isin(X_test.index)].Date, columns = ['Date'])
res_eval_df_probs_rfc['DF_ind'] = y_test
res_eval_df_probs_rfc['DF_pred'] = pred_rfc[:,1]


# fig, axs = plt.subplots(len(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date.apply(lambda x: x.year).unique()))
# i = 0
# for year_i in res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
#     data_year_i = res_eval_df_probs[
#         (res_eval_df_probs['DF_ind'] == 1) & (res_eval_df_probs['Date'].apply(lambda x: x.year).isin([year_i]))]
#     axs[i].scatter(data_year_i['Date'], data_year_i['DF_pred'])
#     #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
#     i = i + 1
# plt.show()

# -------

from matplotlib.pyplot import figure
import matplotlib

installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power[installed_capacity_factor_solar_pv_power['Date'].dt.year.isin(range(2007, 2022))].reset_index().drop(columns = 'index')
installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power[['Date', 'DE']]

installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons[installed_capacity_factor_wind_power_ons['Date'].dt.year.isin(range(2007, 2022))].reset_index().drop(columns = 'index')
installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons[['Date', 'DE']]

installed_capacity_factor_wind_power_ons_plus_solar = pd.DataFrame(installed_capacity_factor_wind_power_ons['Date'],  columns= ['Date'])
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.merge(installed_capacity_factor_solar_pv_power, on = 'Date', how = 'left')
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.rename({'DE': 'solar_DE'}, axis = 'columns')
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.merge(installed_capacity_factor_wind_power_ons, on = 'Date', how = 'left')
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.rename({'DE': 'wind_ons_DE'}, axis = 'columns')
installed_capacity_factor_wind_power_ons_plus_solar['sum'] = installed_capacity_factor_wind_power_ons_plus_solar['solar_DE'] + installed_capacity_factor_wind_power_ons_plus_solar['wind_ons_DE']


#plt.scatter(res_eval_df_probs_rfc.reset_index().iloc[1618:]['DF_pred'], installed_capacity_factor_wind_power_ons_plus_solar['sum'].iloc[8760:])

# matplotlib.rc('xtick', labelsize=13)
# matplotlib.rc('ytick', labelsize=13)
# plt.ylim(bottom=0)
# plt.show()

i = 0
j = 0
k = 0
fig, ax = plt.subplots(2,7, figsize=(15, 7), dpi=90)
for year_i in res_eval_df_probs_rfc[res_eval_df_probs_rfc['DF_ind'] == 1].Date.apply(lambda x: x.year).unique()[1:]:
    data_sum_year_i = installed_capacity_factor_wind_power_ons_plus_solar[
        installed_capacity_factor_wind_power_ons_plus_solar['Date'].apply(lambda x: x.year).isin([year_i])]
    data_year_i = res_eval_df_probs_rfc[res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i])]
    data_year_i_df = res_eval_df_probs_rfc[
        (res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
    data_all = data_sum_year_i.merge(data_year_i, how = 'left', on = 'Date')
    #ax[j, k].scatter(data_year_i['DF_pred'],
    #            data_sum_year_i['sum'])
    ax[j, k].scatter(data_all['DF_pred'],
                data_all['wind_ons_DE'], s=3, color='indigo', label=str(year_i))
    ax[j, k].hlines(y = 0.25,xmin=0, xmax=1, colors = 'crimson')
    ax[j, k].set_xlim(left=0, right=1)
    ax[j, k].set_ylim(bottom=0, top=4.5)
    ax[j, k].legend(loc="upper right")
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    #ax[j,k].ylim(bottom=0)
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    if k <= 5:
        k = k + 1
    else:
        k = 0
    if i >= 6:
        j = 1
    i = i + 1

fig.supxlabel('Probability (for onshore wind CF <= 0.25) Classifier')
fig.supylabel('Seasonal adjusted and corrected onshore wind capacity factor')
plt.savefig(
    'ClassificationResults_Probabilities_ons_wind_Scatterplot_per_year.png')
plt.show()

i = 0
for year_i in res_eval_df_probs_rfc[res_eval_df_probs_rfc['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    fig, ax = plt.subplots(2, figsize=(23, 7), dpi=90)
    data_wind_year_i = installed_capacity_factor_wind_power_ons[
        installed_capacity_factor_wind_power_ons['Date'].apply(lambda x: x.year).isin([year_i])]
    data_solar_year_i = installed_capacity_factor_solar_pv_power[
        installed_capacity_factor_solar_pv_power['Date'].apply(lambda x: x.year).isin([year_i])]
    data_sum_year_i = installed_capacity_factor_wind_power_ons_plus_solar[
        installed_capacity_factor_wind_power_ons_plus_solar['Date'].apply(lambda x: x.year).isin([year_i])]
    data_year_i = res_eval_df_probs_rfc[res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i])]
    data_year_i_df = res_eval_df_probs_rfc[
        (res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
    # data_year_i_df = res_eval_df_probs_rfc[(res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax[0].plot(data_wind_year_i['Date'], data_wind_year_i['DE'], color='navy', label='Onshore wind adjusted CF')
    #ax[1].plot(data_solar_year_i['Date'], data_solar_year_i['DE'], color='indigo', label='Solar adjusted CF')
    #ax[2].plot(data_sum_year_i['Date'], data_sum_year_i['sum'], color='violet',
    #           label='Sum (Onshore wind + solar) adjusted CFs')
    ax[1].plot(data_year_i['Date'], data_year_i['DF_pred'], color='teal', label='Probability Classifier (Ons Wind)')
    ax[1].scatter(data_year_i_df['Date'], data_year_i_df['DF_pred'], s=7, color='crimson', label='Low ons wind CF', zorder=1)
    # ax = plt.scatter(data_year_i_df['Date'], data_year_i_df['DF_pred'])
    ax[0].hlines(0.25, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    #ax[1].hlines(0.5, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    #ax[2].hlines(1, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    fig.legend()
    plt.savefig(
        'ClassificationResults_Probabilities_OnsWind_CF_vs_classified_probability' + str(year_i) + '.png')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    # plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()
