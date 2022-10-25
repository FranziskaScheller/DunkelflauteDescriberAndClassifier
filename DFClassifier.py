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
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

### Create MasterTable ###
msl_aggr_DE_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland.csv', sep=';')
msl_aggr_DE_79to21['Dates'] = msl_aggr_DE_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_DE_79to21 = msl_aggr_DE_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp mean DE', 'std': 'mslp std DE'})

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
msl_aggr_NL_79to21 = msl_aggr_NL_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp mean NL', 'std': 'mslp std NL'})

msl_aggr_PL_79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska79to90.csv', sep = ';', encoding='latin1')
msl_aggr_PL_91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska91to00.csv', sep = ';', encoding='latin1')
msl_aggr_PL_01to10 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska01to10.csv', sep = ';', encoding='latin1')
msl_aggr_PL_11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslPolska11to21.csv', sep = ';', encoding='latin1')

msl_aggr_PL_79to21 = pd.concat([msl_aggr_PL_79to90, msl_aggr_PL_91to00, msl_aggr_PL_01to10, msl_aggr_PL_11to21])

msl_aggr_PL_79to21['Dates'] = msl_aggr_PL_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_PL_79to21 = msl_aggr_PL_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp mean PL', 'std': 'mslp std PL'})

msl_aggr_FR_79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance79to90.csv', sep = ';', encoding='latin1')
msl_aggr_FR_91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance91to00.csv', sep = ';', encoding='latin1')
msl_aggr_FR_01to10 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance01to10.csv', sep = ';', encoding='latin1')
msl_aggr_FR_11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance11to21.csv', sep = ';', encoding='latin1')

msl_aggr_FR_79to21 = pd.concat([msl_aggr_FR_79to90, msl_aggr_FR_91to00, msl_aggr_FR_01to10, msl_aggr_FR_11to21])

msl_aggr_FR_79to21['Dates'] = msl_aggr_FR_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_FR_79to21 = msl_aggr_FR_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp mean FR', 'std': 'mslp std FR'})

t2m_aggr_DE_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_t2mDeutschland.csv', sep=';')
t2m_aggr_DE_79to21['Dates'] = t2m_aggr_DE_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
t2m_aggr_DE_79to21 = t2m_aggr_DE_79to21.rename(columns = {'Dates': 'Date', 'mean': 't2m mean DE', 'std': 't2m std DE'})

dunkelflaute_dates_DE = pd.read_csv(
    'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('DE') + str(
        '0.5') + '_PVOnshoreWind_AC_dates.csv')

dunkelflaute_dates_DE['DFDates'] = dunkelflaute_dates_DE['0'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
DF_indices = msl_aggr_DE_79to21['Date'].isin(dunkelflaute_dates_DE['DFDates'])
DF_indices_0_1_encoding = DF_indices.apply(lambda x: int(x))
mastertableDFclassifier = msl_aggr_DE_79to21.merge(t2m_aggr_DE_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_GWL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_PL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_NL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_FR_79to21, on = 'Date', how = 'left')
#mastertableDFclassifier['DF_Indicator_h'] = DF_indices_0_1_encoding
mastertableDFclassifier['DF_Indicator'] = DF_indices_0_1_encoding
mastertableDFclassifier['Month'] = mastertableDFclassifier['Date'].apply(lambda x: x.month)
mastertableDFclassifier['Hour'] = mastertableDFclassifier['Date'].apply(lambda x: x.hour)
ohe = OneHotEncoder().fit_transform(X = pd.DataFrame(mastertableDFclassifier['Date'].apply(lambda x: x.month).values)).toarray()
mastertableDFclassifier[['Ind Jan', 'Ind Feb', 'Ind Mar', 'Ind Apr', 'Ind May', 'Ind Jun', 'Ind Jul', 'Ind Aug', 'Ind Sep', 'Ind Oct', 'Ind Nov', 'Ind Dec']] = ohe
# --- Add Lags ---
daily_mean = mastertableDFclassifier.set_index('Date')[['mslp mean DE', 'mslp std DE']].groupby(pd.Grouper(freq='d')).mean().rename({'mslp mean DE': 'mslp daily mean DE', 'mslp std DE': 'mslp daily std DE' }, axis = 'columns').reset_index()
mastertableDFclassifier = mastertableDFclassifier.merge(daily_mean, on = 'Date', how = 'left')

daily_max = mastertableDFclassifier.set_index('Date')[['mslp mean DE', 'mslp std DE']].groupby(pd.Grouper(freq='d')).max().rename({'mslp mean DE': 'mslp daily max of mean DE', 'mslp std DE': 'mslp daily max of std DE' }, axis = 'columns').reset_index()
mastertableDFclassifier = mastertableDFclassifier.merge(daily_max, on = 'Date', how = 'left')

daily_min = mastertableDFclassifier.set_index('Date')[['mslp mean DE', 'mslp std DE']].groupby(pd.Grouper(freq='d')).min().rename({'mslp mean DE': 'mslp daily min of mean DE', 'mslp std DE': 'mslp daily min of std DE' }, axis = 'columns').reset_index()
mastertableDFclassifier = mastertableDFclassifier.merge(daily_min, on = 'Date', how = 'left')

mastertableDFclassifier[['mslp daily mean DE', 'mslp daily std DE', 'mslp daily max of mean DE', 'mslp daily max of std DE', 'mslp daily min of mean DE', 'mslp daily min of std DE']] = mastertableDFclassifier[['mslp daily mean DE', 'mslp daily std DE', 'mslp daily max of mean DE', 'mslp daily max of std DE', 'mslp daily min of mean DE', 'mslp daily min of std DE']].fillna(method="ffill")

mastertableDFclassifier['mslp mean DE lag 1h'] = mastertableDFclassifier['mslp mean DE'].shift(1)
mastertableDFclassifier['mslp mean DE lag 2h'] = mastertableDFclassifier['mslp mean DE'].shift(2)
mastertableDFclassifier['mslp mean DE lag 5h'] = mastertableDFclassifier['mslp mean DE'].shift(5)
mastertableDFclassifier['mslp mean DE lag 24h'] = mastertableDFclassifier['mslp mean DE'].shift(24)
mastertableDFclassifier['mslp mean DE lag 48h'] = mastertableDFclassifier['mslp mean DE'].shift(48)
mastertableDFclassifier['mslp mean DE lag 120h'] = mastertableDFclassifier['mslp mean DE'].shift(120)

mastertableDFclassifier['mslp mean PL lag 24h'] = mastertableDFclassifier['mslp mean PL'].shift(24)
mastertableDFclassifier['mslp mean NL lag 24h'] = mastertableDFclassifier['mslp mean NL'].shift(24)
mastertableDFclassifier['mslp mean FR lag 24h'] = mastertableDFclassifier['mslp mean FR'].shift(24)

mastertableDFclassifier['mslp mean PL lag 120h'] = mastertableDFclassifier['mslp mean PL'].shift(120)
mastertableDFclassifier['mslp mean NL lag 120h'] = mastertableDFclassifier['mslp mean NL'].shift(120)
mastertableDFclassifier['mslp mean FR lag 120h'] = mastertableDFclassifier['mslp mean FR'].shift(120)

mastertableDFclassifier['mslp std DE lag 1h'] = mastertableDFclassifier['mslp std DE'].shift(1)
mastertableDFclassifier['mslp std DE lag 2h'] = mastertableDFclassifier['mslp std DE'].shift(2)
mastertableDFclassifier['mslp std DE lag 5h'] = mastertableDFclassifier['mslp std DE'].shift(5)
mastertableDFclassifier['mslp std DE lag 24h'] = mastertableDFclassifier['mslp std DE'].shift(24)
mastertableDFclassifier['mslp std DE lag 48h'] = mastertableDFclassifier['mslp std DE'].shift(48)
mastertableDFclassifier['mslp std DE lag 120h'] = mastertableDFclassifier['mslp std DE'].shift(120)

mastertableDFclassifier['mslp std PL lag 24h'] = mastertableDFclassifier['mslp std PL'].shift(24)
mastertableDFclassifier['mslp std NL lag 24h'] = mastertableDFclassifier['mslp std NL'].shift(24)
mastertableDFclassifier['mslp std FR lag 24h'] = mastertableDFclassifier['mslp std FR'].shift(24)

mastertableDFclassifier['mslp std PL lag 120h'] = mastertableDFclassifier['mslp std PL'].shift(120)
mastertableDFclassifier['mslp std NL lag 120h'] = mastertableDFclassifier['mslp std NL'].shift(120)
mastertableDFclassifier['mslp std FR lag 120h'] = mastertableDFclassifier['mslp std FR'].shift(120)

mastertableDFclassifier['t2m mean DE lag 24h'] = mastertableDFclassifier['t2m mean DE'].shift(24)
mastertableDFclassifier['t2m mean DE lag 48h'] = mastertableDFclassifier['t2m mean DE'].shift(48)
mastertableDFclassifier['t2m mean DE lag 120h'] = mastertableDFclassifier['t2m mean DE'].shift(120)

mastertableDFclassifier['t2m std DE lag 24h'] = mastertableDFclassifier['t2m std DE'].shift(24)
mastertableDFclassifier['t2m std DE lag 48h'] = mastertableDFclassifier['t2m std DE'].shift(48)
mastertableDFclassifier['t2m std DE lag 120h'] = mastertableDFclassifier['t2m std DE'].shift(120)

for col in msl_aggr_GWL_79to21.columns[1:-1]:
    mastertableDFclassifier[str(col) + '_Lag_24h'] = mastertableDFclassifier[str(col)].shift(24)
    mastertableDFclassifier[str(col) + '_Lag_120h'] = mastertableDFclassifier[str(col)].shift(120)

mastertableDFclassifier = mastertableDFclassifier.rename(columns = {'mean_Greenland': 'mslp mean Greenland', 'std_Greenland': 'mslp std Greenland', 'mean_Iceland': 'mslp mean Iceland', 'std_Iceland': 'mslp std Iceland', 'mean_British_Isles': 'mslp mean British Isles', 'std_British_Isles': 'mslp std British Isles', 'mean_Mediterranean_Sea': 'mslp mean Mediterranean Sea', 'std_Mediterranean_Sea': 'mslp std Mediterranean Sea', 'mean_Sea_west_Iberian_Peninsula': 'mslp mean Sea west Iberian Peninsula', 'std_Sea_west_Iberian_Peninsula': 'mslp std Sea west Iberian Peninsula', 'mean_Norwegian_Sea': 'mslp mean Norwegian Sea', 'std_Norwegian_Sea': 'mslp std Norwegian Sea', 'mean_North_Sea': 'mslp mean North Sea', 'std_North_Sea': 'mslp std North Sea', 'mean_Western_Russia': 'mslp mean Western Russia', 'std_Western_Russia': 'mslp std Western Russia', 'mean_Sweden': 'mslp mean Sweden', 'std_Sweden': 'mslp std Sweden'})

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

# X = mastertableDFclassifier[['mslp mean DE', 'mslp mean DE lag 5h', 'mslp mean DE lag 24h', 'mslp mean DE lag 48h', 'mslp mean DE lag 120h',
#                              'mslp std DE', 'mslp std DE lag 5h', 'mslp std DE lag 24h', 'mslp std DE lag 48h', 'mslp std DE lag 120h',
#                              't2m mean DE', 't2m mean DE lag 24h', 't2m mean DE lag 48h', 't2m mean DE lag 120h',
#                              't2m std DE', 't2m std DE lag 24h', 't2m std DE lag 48h', 't2m std DE lag 120h',
#                             'mslp daily mean DE', 'mslp daily std DE', 'mslp daily max of mean DE', 'mslp daily max of std DE', 'mslp daily min of mean DE', 'mslp daily min of std DE',
#                             'mslp mean PL', 'mslp std PL', 'mslp mean NL', 'mslp std NL', 'mslp mean FR', 'mslp std FR',
#                             'mslp mean PL lag 24h', 'mslp std PL lag 24h', 'mslp mean NL lag 24h', 'mslp std NL lag 24h', 'mslp mean FR lag 24h', 'mslp std FR lag 24h',
#                              'mslp mean PL lag 120h', 'mslp std PL lag 120h', 'mslp mean NL lag 120h','mslp std NL lag 120h', 'mslp mean FR lag 120h', 'mslp std FR lag 120h',
#                              'mslp mean Greenland' , 'mslp mean Iceland' , 'mslp mean British Isles',  'mslp mean Mediterranean Sea', 'mslp mean Sea west Iberian Peninsula','mslp mean Norwegian Sea' , 'mslp mean North Sea', 'mslp mean Western Russia',  'mslp mean Sweden',
#                              'mslp std Greenland', 'mslp std Iceland', 'mslp std British Isles',
#                              'mslp std Mediterranean Sea', 'mslp std Sea west Iberian Peninsula',
#                              'mslp std Norwegian Sea', 'mslp std North Sea', 'mslp std Western Russia',
#                              'mslp std Sweden',
#                              'Ind Jan', 'Ind Feb', 'Ind Mar', 'Ind Apr', 'Ind May', 'Ind Jun', 'Ind Jul', 'Ind Aug', 'Ind Sep', 'Ind Oct', 'Ind Nov', 'Ind Dec']]

X = mastertableDFclassifier[['mslp mean DE', 'mslp mean DE lag 5h', 'mslp mean DE lag 24h', 'mslp mean DE lag 48h', 'mslp mean DE lag 120h',
                             'mslp std DE', 'mslp std DE lag 5h', 'mslp std DE lag 24h', 'mslp std DE lag 48h', 'mslp std DE lag 120h',
                             't2m mean DE', 't2m mean DE lag 24h', 't2m mean DE lag 48h', 't2m mean DE lag 120h',
                            'mslp daily mean DE', 'mslp daily std DE', 'mslp daily max of mean DE', 'mslp daily max of std DE', 'mslp daily min of mean DE', 'mslp daily min of std DE',
                            'mslp mean PL', 'mslp std PL', 'mslp mean NL', 'mslp std NL', 'mslp mean FR', 'mslp std FR',
                            'mslp mean PL lag 24h', 'mslp std PL lag 24h', 'mslp mean NL lag 24h', 'mslp std NL lag 24h', 'mslp mean FR lag 24h', 'mslp std FR lag 24h',
                             'mslp mean PL lag 120h', 'mslp std PL lag 120h', 'mslp mean NL lag 120h','mslp std NL lag 120h', 'mslp mean FR lag 120h', 'mslp std FR lag 120h',
                             'mslp mean Greenland' , 'mslp mean Iceland' , 'mslp mean British Isles',  'mslp mean Mediterranean Sea', 'mslp mean Sea west Iberian Peninsula','mslp mean Norwegian Sea' , 'mslp mean North Sea', 'mslp mean Western Russia',  'mslp mean Sweden',
                             'mslp std Sweden',
                             'Ind Jan', 'Ind Feb', 'Ind Mar', 'Ind Apr', 'Ind May', 'Ind Jun', 'Ind Jul', 'Ind Aug', 'Ind Sep', 'Ind Oct', 'Ind Nov', 'Ind Dec']]

y = mastertableDFclassifier['DF_Indicator']

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


from numpy import sqrt
from numpy import argmax

from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.metrics import roc_curve

brf200 = BalancedRandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
pred_brf200 = brf200.predict_proba(X_test)
#
brf400 = BalancedRandomForestClassifier(n_estimators=400, random_state=0).fit(X_train, y_train)
pred_brf400 = brf400.predict_proba(X_test)
#
rfc200 = RandomForestClassifier(random_state=0, n_estimators= 200).fit(X_train, y_train)
pred_rfc200 = rfc200.predict_proba(X_test)
rfc400 = RandomForestClassifier(random_state=0, n_estimators= 400).fit(X_train, y_train)
pred_rfc400 = rfc400.predict_proba(X_test)
#
# # calculate roc curves
# red_names = [pred_brf200, pred_brf400, pred_rfc200, pred_rfc400]
# red_labels = ['BRF 200 Trees', 'BRF 400 Trees', 'RF 200 Trees', 'RF 400 Trees']
# fig, ax = plt.subplots(2, 2, figsize=(16, 16), dpi=120)
# k = 0
# for i in range(0, 4):
#     fpr, tpr, thresholds = roc_curve(y_test, red_names[i][:, 1])
#     # calculate the g-mean for each threshold
#     gmeans = sqrt(tpr * (1 - fpr))
#     # locate the index of the largest g-mean
#     ix = argmax(gmeans)
#     print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
#     weighted_acc = 0.8 * tpr + 0.2 * (1 - fpr)
#     # locate the index of the largest g-mean
#     ix_weighted_acc = argmax(weighted_acc)
#     print('Best Threshold=%f, weighted_acc=%.3f' % (thresholds[ix_weighted_acc], weighted_acc[ix_weighted_acc]))
#     print('AUC: ' + str( roc_auc_score(y_test, red_names[i][:, 1])))
#
#     # plot the roc curve for the model
#     plt.subplot(2, 2, (i + 1))
#     plt.title(str(red_labels[k]), fontsize=24)
#     plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill', color='mediumaquamarine')
#     plt.plot(fpr, tpr, marker='.', label='ROC curve', color='teal')
#     plt.scatter(fpr[ix], tpr[ix], marker='o', s=80, color='indigo',
#                 label='Best G-Mean ' + str(np.round(max(gmeans), 3)))
#     plt.scatter(fpr[ix_weighted_acc], tpr[ix_weighted_acc], marker='o', s=80, color='mediumorchid',
#                 label='Best Weighted Accuracy ' + str(np.round(max(weighted_acc), 3)))
#     # axis labels
#     plt.xlabel('1 - True Negative Rate', fontsize=20)
#     plt.ylabel('True Positive Rate', fontsize=20)
#     plt.xticks(fontsize=19)
#     plt.yticks(fontsize=19)
#     plt.legend(fontsize=18)
#     k = k + 1
#     plt.tight_layout()
#     plt.savefig(
#         'ROCCurves.png')
#     # show the plot
#
# pyplot.show()
#
red_names = [[pred_brf200, pred_brf400], [pred_rfc200, pred_rfc400]]
red_labels = ['Balanced Random Forest with 200 and 400 trees', 'Random Forest with 200 and 400 trees']
fig, ax = plt.subplots(2, 1, figsize=(12, 26), dpi=120)
k = 0
for i in range(0, 2):
    fpr_200, tpr_200, thresholds_200 = roc_curve(y_test, red_names[i][0][:, 1])
    fpr_400, tpr_400, thresholds_400 = roc_curve(y_test, red_names[i][1][:, 1])
    # calculate the g-mean for each threshold
    gmeans_200 = sqrt(tpr_200 * (1 - fpr_200))
    gmeans_400 = sqrt(tpr_400 * (1 - fpr_400))
    # locate the index of the largest g-mean
    ix_200 = argmax(gmeans_200)
    ix_400 = argmax(gmeans_400)
    print('Best Threshold200=%f, G-Mean400=%.3f' % (thresholds_200[ix_200], gmeans_200[ix_200]))
    print('Best Threshold400=%f, G-Mean400=%.3f' % (thresholds_400[ix_400], gmeans_400[ix_400]))
    weighted_acc_200 = 0.666 * tpr_200 + (1-0.666) * (1 - fpr_200)
    # locate the index of the largest g-mean
    ix_weighted_acc_200 = argmax(weighted_acc_200)
    print('Best Threshold200=%f, weighted_acc200=%.3f' % (thresholds_200[ix_weighted_acc_200], weighted_acc_200[ix_weighted_acc_200]))
    print('AUC200: ' + str( roc_auc_score(y_test, red_names[i][0][:, 1])))
    weighted_acc_400 = 0.666 * tpr_400 + (1-0.666) * (1 - fpr_400)
    # locate the index of the largest g-mean
    ix_weighted_acc_400 = argmax(weighted_acc_400)
    print('Best Threshold_400=%f, weighted_acc_400=%.3f' % (thresholds_400[ix_weighted_acc_400], weighted_acc_400[ix_weighted_acc_400]))
    print('AUC_400: ' + str( roc_auc_score(y_test, red_names[i][1][:, 1])))
    # plot the roc curve for the model
    plt.subplot(2, 1, (i + 1))
    plt.title(str(red_labels[k]), fontsize=30)
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill', color='mediumaquamarine', linewidth=3.5, zorder=-1)
    plt.plot(fpr_200, tpr_200, marker='.', label='ROC curve with 200 trees', color='teal', linewidth=3.5, zorder=-1)
    plt.plot(fpr_400, tpr_400, marker='.', label='ROC curve with 400 trees', color='darkslategrey', linewidth=3.5, zorder=-1)
    plt.scatter(fpr_200[ix_200], tpr_200[ix_200], marker='o', s=160, color='indigo',
                label='Best G-Mean with 200 trees ' + str(np.round(max(gmeans_200), 3)), zorder=1)
    plt.scatter(fpr_200[ix_weighted_acc_200], tpr_200[ix_weighted_acc_200], marker='o', s=160, color='mediumorchid',
                label='Best Weighted Accuracy with 200 trees ' + str(np.round(max(weighted_acc_200), 3)), zorder=1)
    plt.scatter(fpr_400[ix_400], tpr_400[ix_400], marker='o', s=160, color='gold',
                label='Best G-Mean with 400 trees ' + str(np.round(max(gmeans_400), 3)), zorder=1)
    plt.scatter(fpr_400[ix_weighted_acc_400], tpr_400[ix_weighted_acc_400], marker='o', s=160, color='orange',
                label='Best Weighted Accuracy with 400 trees ' + str(np.round(max(weighted_acc_400), 3)), zorder=1)
   # axis labels
    plt.xlabel('1 - True Negative Rate', fontsize=27)
    plt.ylabel('True Positive Rate', fontsize=27)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=22)
    k = k + 1
    plt.tight_layout()
    plt.savefig(
        'ROCCurves.png')
    # show the plot
pyplot.show()
#



rfc = RandomForestClassifier(random_state=0, n_estimators= 400).fit(X_train, y_train)
pred_rfc = rfc.predict_proba(X_test)
#
clf_pred_m_rfc = pd.DataFrame(pred_rfc)
clf_pred_m_rfc['real'] = y_test.values
#
clf_res_y_rfc = np.zeros(len(y_test))
clf_res_y_rfc[pred_rfc[:,1] >= 0.0025] = 1
clf_res_y_rfc = pd.DataFrame(clf_res_y_rfc)
#
conf_mat_logreg_rfc = confusion_matrix(y_test, clf_res_y_rfc.values)
precision_logreg_rfc = precision_score(y_test, clf_res_y_rfc.values)
recall_logreg_rfc = recall_score(y_test, clf_res_y_rfc.values)

#brf = BalancedRandomForestClassifier(n_estimators=50, random_state=0)
brf = BalancedRandomForestClassifier(n_estimators=400, random_state=0).fit(X_train, y_train)
pred_brf = brf.predict_proba(X_test)
brf_feature_imp = brf.feature_importances_
forest_importances = pd.Series(brf_feature_imp, index=X.columns)
forest_importances = forest_importances.sort_values()

# std = np.std([tree.feature_importances_ for tree in brf.estimators_], axis=0)
# fig, ax = plt.subplots(figsize=(16, 25), dpi=100)
# forest_importances.plot.barh(ax=ax, color = 'teal')
# ax.set_title("Feature importances",  fontsize=33)
# plt.rc('xtick', labelsize=23)
# plt.rc('ytick', labelsize=23)
# fig.tight_layout()
# plt.savefig(
#    'Feature_importance_all.png')
# fig.show()
#
clf_pred_m_brf = pd.DataFrame(pred_brf)
clf_pred_m_brf['real'] = y_test.values
#
clf_res_y_brf = np.zeros(len(y_test))
clf_res_y_brf[pred_brf[:,1] >= 0.198] = 1
clf_res_y_brf = pd.DataFrame(clf_res_y_brf)
#
conf_mat_logreg_brf = confusion_matrix(y_test, clf_res_y_brf.values)
precision_logreg_brf = precision_score(y_test, clf_res_y_brf.values)
recall_logreg_brf = recall_score(y_test, clf_res_y_brf.values)
target_names = ['NoDF', 'DF']  # doctest : +NORMALIZE_WHITESPACE
res_brt = classification_report_imbalanced(y_test.values, clf_res_y_brf.values, target_names=target_names)
res_rf = classification_report_imbalanced(y_test.values, clf_res_y_rfc.values, target_names=target_names)

dummy_clf = DummyClassifier(strategy="stratified").fit(X_train, y_train)
pred_dummy_clf = dummy_clf.predict_proba(X_test)

# r_auc_sc = roc_auc_score(y_test, pred_brf[:, 1])
# from numpy import sqrt
# from numpy import argmax
#
# from sklearn.linear_model import LogisticRegression
# from matplotlib import pyplot
# from sklearn.metrics import roc_curve
# # calculate roc curves
# fpr, tpr, thresholds = roc_curve(y_test, pred_brf[:, 1])
# # calculate the g-mean for each threshold
# gmeans = sqrt(tpr * (1-fpr))
# # locate the index of the largest g-mean
# ix = argmax(gmeans)
# print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# weighted_acc = 0.8*tpr + 0.2*(1-fpr)
# # locate the index of the largest g-mean
# ix_weighted_acc = argmax(weighted_acc)
# print('Best Threshold=%f, weighted_acc=%.3f' % (thresholds[ix], weighted_acc[ix]))
# # plot the roc curve for the model
# pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
# pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best_G-Mean')
# pyplot.scatter(fpr[ix_weighted_acc], tpr[ix_weighted_acc], marker='o', color='red', label='Best_WeightedAcc')
# # axis labels
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# pyplot.legend()
# # show the plot
# pyplot.show()
#
# weighted_acc = 0.8*tpr + 0.2*(1-fpr)
# # locate the index of the largest g-mean
# ix_weighted_acc = argmax(weighted_acc)
# print('Best Threshold=%f, weighted_acc=%.3f' % (thresholds[ix], weighted_acc[ix]))
# # plot the roc curve for the model
# pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
# pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# # axis labels
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# pyplot.legend()
# # show the plot
# pyplot.show()
#
#
# # ---- Evaluation ----
# dummy_clf = DummyClassifier(strategy="stratified").fit(X_train, y_train)
# pred_dummy_clf = dummy_clf.predict_proba(X_test)
#
# dummy_clf = DummyClassifier(strategy="stratified")
# scoring = ["accuracy", "balanced_accuracy"]
# cv_result = cross_validate(dummy_clf, df_res, y_test.values, scoring=scoring)
# index = []
# scores = {"Accuracy": [], "Balanced accuracy": []}
# index += ["Dummy classifier"]
# cv_result = cross_validate(dummy_clf, df_res, y_test.values, scoring=scoring)
# scores["Accuracy"].append(cv_result["test_accuracy"].mean())
# scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
#
# df_scores = pd.DataFrame(scores, index=index)
# #df_scores
#
# clf = BaggingClassifier(base_estimator=SVC(),n_estimators=150, random_state=0).fit(X_train, y_train)
#
# #rfc = RandomForestClassifier(random_state=0, min_samples_leaf = 2, n_estimators= 200).fit(X_train, y_train)
# pred_clf = clf.predict_proba(X_test)
# #
# clf_pred_m_clf = pd.DataFrame(pred_clf)
# clf_pred_m_clf['real'] = y_test.values
# #
# clf_res_y_clf = np.zeros(len(y_test))
# clf_res_y_clf[pred_clf[:,1] >= 0.006] = 1
# clf_res_y_clf = pd.DataFrame(clf_res_y_clf)
# #
# conf_mat_logreg_clf = confusion_matrix(y_test, clf_res_y_clf.values)
# precision_logreg_clf = precision_score(y_test, clf_res_y_clf.values)
# recall_logreg_clf = recall_score(y_test, clf_res_y_clf.values)
#
#
# clf = LogisticRegression(random_state=0, max_iter=600).fit(X_train, y_train)
# pred_cfl = clf.predict_proba(X_test)
# clf_pred_m = pd.DataFrame(pred_cfl)
# clf_pred_m['real'] = y_test.values
#
# clf_res_y = np.zeros(len(y_test))
# clf_res_y[pred_cfl[:,1] >= 0.006] = 1
# clf_res_y = pd.DataFrame(clf_res_y)
#
# conf_mat_logreg = confusion_matrix(y_test, clf_res_y.values)
# precision_logreg = precision_score(y_test, clf_res_y.values)
# recall_logreg = recall_score(y_test, clf_res_y.values)

#---
# res_eval_df_probs = pd.DataFrame(mastertableDFclassifier[mastertableDFclassifier.index.isin(X_test.index)].Date, columns = ['Date'])
# res_eval_df_probs['DF_ind'] = y_test
# res_eval_df_probs['DF_pred'] = pred_cfl[:,1]

clf_res_y_brf = np.zeros(len(y_test))
clf_res_y_brf[pred_brf[:,1] >= 0.198] = 1
clf_res_y_brf = pd.DataFrame(clf_res_y_brf)

res_eval_df_probs_rfc = pd.DataFrame(mastertableDFclassifier[mastertableDFclassifier.index.isin(X_test.index)].Date, columns = ['Date'])
res_eval_df_probs_rfc['DF_ind'] = y_test
res_eval_df_probs_rfc['DF_pred'] = pred_brf[:,1]


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

plt.scatter(res_eval_df_probs_rfc.reset_index().iloc[1618:]['DF_pred'], installed_capacity_factor_wind_power_ons_plus_solar['sum'].iloc[8760:])

matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)
plt.ylim(bottom=0)
plt.show()

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
                data_all['sum'], s=3, color='indigo', label=str(year_i))

    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    fig.legend()
    #ax[j,k].ylim(bottom=0)
    # plt.savefig(
    #    'ClassificationResults_Probabilities_all_rfc_incl_CFs' + str(year_i) + '.png')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    # plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    if k <= 5:
        k = k + 1
    else:
        k = 0
    if i >= 6:
        j = 1
    i = i + 1

plt.show()



# i = 0
# for year_i in res_eval_df_probs_rfc[res_eval_df_probs_rfc['DF_ind'] == 1].Date.apply(lambda x: x.year).unique()[1:]:
#     data_sum_year_i = installed_capacity_factor_wind_power_ons_plus_solar[
#         installed_capacity_factor_wind_power_ons_plus_solar['Date'].apply(lambda x: x.year).isin([year_i])]
#     data_year_i = res_eval_df_probs_rfc[res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i])]
#     data_year_i_df = res_eval_df_probs_rfc[
#         (res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
#
#     plt.scatter(data_sum_year_i['sum'].rank(pct=True), data_year_i['DF_pred'].rank(pct=True))
#
#     matplotlib.rc('xtick', labelsize=13)
#     matplotlib.rc('ytick', labelsize=13)
#     plt.savefig(
#     'ClassificationResults_Probabilities_all_rfc_incl_CFs' + str(year_i) + '.png')
#     #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#     plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
#     i = i + 1
#     plt.show()

i = 0
for year_i in res_eval_df_probs_rfc[res_eval_df_probs_rfc['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    fig, ax = plt.subplots(4, figsize=(23, 9), dpi=90)
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
    ax[0].plot(data_wind_year_i['Date'], data_wind_year_i['DE'], color='navy', label='Adjusted onshore wind CF')
    ax[1].plot(data_solar_year_i['Date'], data_solar_year_i['DE'], color='indigo', label='Adjusted solar CF')
    ax[2].plot(data_sum_year_i['Date'], data_sum_year_i['sum'], color='violet',
               label='Sum of adjusted onshore wind and solar CF')
    ax[3].plot(data_year_i['Date'], data_year_i['DF_pred'], color='teal', label='Probability BRF classifier')
    ax[3].scatter(data_year_i_df['Date'], data_year_i_df['DF_pred'], s=7, color='crimson', label='Dunkelflaute', zorder=1)
    # ax = plt.scatter(data_year_i_df['Date'], data_year_i_df['DF_pred'])
    ax[0].hlines(0.5, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    ax[1].hlines(0.5, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    ax[2].hlines(1, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    fig.legend()
    #plt.savefig(
    #    'ClassificationResults_Probabilities_all_rfc_incl_CFs' + str(year_i) + '.png')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    # plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()

#----

i = 0
for year_i in res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    figure(figsize=(30, 6), dpi=80)
    data_year_i = res_eval_df_probs[
        (res_eval_df_probs['DF_ind'] == 1) & (res_eval_df_probs['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax = plt.scatter(data_year_i['Date'], data_year_i['DF_pred'])
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    #plt.savefig(
    #    'ClassificationResults_Probabilities_' + str(year_i) + '.png')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()

i = 0
for year_i in res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    figure(figsize=(30, 6), dpi=80)
    data_year_i = res_eval_df_probs[res_eval_df_probs['Date'].apply(lambda x: x.year).isin([year_i])]
    data_year_i_df = res_eval_df_probs[(res_eval_df_probs['DF_ind'] == 1) & (res_eval_df_probs['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax = plt.scatter(data_year_i['Date'], data_year_i['DF_pred'])
    ax = plt.scatter(data_year_i_df['Date'], data_year_i_df['DF_pred'])
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    #plt.savefig(
    #    'ClassificationResults_Probabilities_all_' + str(year_i) + '.png')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()

i = 0
for year_i in res_eval_df_probs_rfc[res_eval_df_probs_rfc['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    figure(figsize=(30, 6), dpi=80)
    data_year_i = res_eval_df_probs_rfc[
        (res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax = plt.scatter(data_year_i['Date'], data_year_i['DF_pred'])
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    #plt.savefig(
    #    'ClassificationResults_Probabilities_rfc_' + str(year_i) + '.png')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()

i = 0
for year_i in res_eval_df_probs_rfc[res_eval_df_probs_rfc['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    figure(figsize=(30, 6), dpi=80)
    data_year_i = res_eval_df_probs_rfc[res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i])]
    data_year_i_df = res_eval_df_probs_rfc[(res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax = plt.scatter(data_year_i['Date'], data_year_i['DF_pred'])
    ax = plt.scatter(data_year_i_df['Date'], data_year_i_df['DF_pred'])
    ax = plt.hlines(0.01, data_year_i['Date'].iloc[0], data_year_i['Date'].iloc[-1], 'green')
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    #plt.savefig(
    #    'ClassificationResults_Probabilities_all_rfc_' + str(year_i) + '.png')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()




#----

res_eval_df = pd.DataFrame(mastertableDFclassifier[mastertableDFclassifier.index.isin(X_test.index)].Date, columns = ['Date'])
res_eval_df['DF_ind'] = y_test
res_eval_df['DF_pred'] = clf_res_y.values

plt.plot(msl_aggr_DE_79to21[msl_aggr_DE_79to21.index.isin(X_test.index)].Date, y_test)
plt.plot(msl_aggr_DE_79to21[msl_aggr_DE_79to21.index.isin(X_test.index)].Date, clf_res_y.values)
plt.show()

plt.plot(res_eval_df[res_eval_df['DF_ind'] == 1].Date, res_eval_df[res_eval_df['DF_ind'] == 1].DF_pred)
plt.show()

FN = res_eval_df[(res_eval_df['DF_ind'] == 1) & (res_eval_df['DF_pred'] == 0)].Date
TP = res_eval_df[(res_eval_df['DF_ind'] == 1) & (res_eval_df['DF_pred'] == 1)].Date
#score_test = knn_fit.score(X_test, y_test)

#cv_score = cross_val_score(neigh, X, y, cv=5)
print(1)
