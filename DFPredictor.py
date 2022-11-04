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
#
# # 24h ahead
#X = X.iloc[0:len(X)-24].reset_index().drop(columns = 'index')
#y = mastertableDFclassifier.iloc[24:].reset_index().drop(columns = 'index')['DF_Indicator']
#
X_24h_ahead = X.iloc[0:len(X)-24].reset_index().drop(columns = 'index')
X_72h_ahead = X.iloc[0:len(X)-72].reset_index().drop(columns = 'index')
X_120h_ahead = X.iloc[0:len(X)-120].reset_index().drop(columns = 'index')
X_168h_ahead = X.iloc[0:len(X)-168].reset_index().drop(columns = 'index')

#
y_24h_ahead = mastertableDFclassifier.iloc[24:].reset_index().drop(columns = 'index')['DF_Indicator']
y_72h_ahead = mastertableDFclassifier.iloc[72:].reset_index().drop(columns = 'index')['DF_Indicator']
y_120h_ahead = mastertableDFclassifier.iloc[120:].reset_index().drop(columns = 'index')['DF_Indicator']
y_168h_ahead = mastertableDFclassifier.iloc[168:].reset_index().drop(columns = 'index')['DF_Indicator']

#
# # Train test split
#
X_train24h_ahead, X_test24h_ahead, y_train24h_ahead, y_test24h_ahead = train_test_split(X_24h_ahead, y_24h_ahead, test_size=0.33, shuffle = False)
X_train72h_ahead, X_test72h_ahead, y_train72h_ahead, y_test72h_ahead = train_test_split(X_72h_ahead, y_72h_ahead, test_size=0.33, shuffle = False)
X_train120h_ahead, X_test120h_ahead, y_train120h_ahead, y_test120h_ahead = train_test_split(X_120h_ahead, y_120h_ahead, test_size=0.33, shuffle = False)
X_train168h_ahead, X_test168h_ahead, y_train168h_ahead, y_test168h_ahead = train_test_split(X_168h_ahead, y_168h_ahead, test_size=0.33, shuffle = False)

# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
#
# print(1)
# # neigh_50 = KNeighborsClassifier(n_neighbors=50)
# # knn_fit_50 = neigh_50.fit(X_train, y_train)
# # knn_pred_50 = knn_fit_50.predict_proba(X_test)
# #
# # knn_pred_m_50 = pd.DataFrame(knn_pred_50)
# # knn_pred_m_50['real'] = y_test.values
# #
# # knn_res_y_50 = np.zeros(len(y_test))
# # knn_res_y_50[knn_pred_50[:,1] >= 0.005] = 1
# # knn_res_y_50 = pd.DataFrame(knn_res_y_50)
# # #
# # conf_mat_50 = confusion_matrix(y_test, knn_res_y_50.values)
# # precision_50 = precision_score(y_test, knn_res_y_50.values)
# # recall_50 = recall_score(y_test, knn_res_y_50.values)
# #
# # neigh_70 = KNeighborsClassifier(n_neighbors=70)
# # knn_fit_70 = neigh_70.fit(X_train, y_train)
# # knn_pred_70 = knn_fit_70.predict_proba(X_test)
# #
# # knn_pred_m_70 = pd.DataFrame(knn_pred_70)
# # knn_pred_m_70['real'] = y_test.values
# #
# # knn_res_y_70 = np.zeros(len(y_test))
# # knn_res_y_70[knn_pred_70[:,1] >= 0.05] = 1
# # knn_res_y_70 = pd.DataFrame(knn_res_y_70)
# #
# # conf_mat_70 = confusion_matrix(y_test, knn_res_y_70.values)
# # precision_70 = precision_score(y_test, knn_res_y_70.values)
# # recall_70 = recall_score(y_test, knn_res_y_70.values)
#
# # neigh_30 = KNeighborsClassifier(n_neighbors=30)
# # knn_fit_30 = neigh_30.fit(X_train, y_train)
# # knn_pred_30 = knn_fit_30.predict_proba(X_test)
# #
# # knn_pred_m_30 = pd.DataFrame(knn_pred_30)
# # knn_pred_m_30['real'] = y_test.values
# #
# # knn_res_y_30 = np.zeros(len(y_test))
# # knn_res_y_30[knn_pred_30[:,1] >= 0.05] = 1
# # knn_res_y_30 = pd.DataFrame(knn_res_y_30)
# #
# # conf_mat_30 = confusion_matrix(y_test, knn_res_y_30.values)
# # precision_30 = precision_score(y_test, knn_res_y_30.values)
# # recall_30 = recall_score(y_test, knn_res_y_30.values)
#
# # neigh_20 = KNeighborsClassifier(n_neighbors=20)
# # knn_fit_20 = neigh_20.fit(X_train, y_train)
# # knn_pred_20 = knn_fit_20.predict_proba(X_test)
# #
# # knn_pred_m_20 = pd.DataFrame(knn_pred_20)
# # knn_pred_m_20['real'] = y_test.values
# #
# # knn_res_y_20 = np.zeros(len(y_test))
# # knn_res_y_20[knn_pred_20[:,1] >= 0.01] = 1
# # knn_res_y_20 = pd.DataFrame(knn_res_y_20)
# #
# # conf_mat_20 = confusion_matrix(y_test, knn_res_y_20.values)
# # precision_20 = precision_score(y_test, knn_res_y_20.values)
# # recall_20 = recall_score(y_test, knn_res_y_20.values)
# #
# # neigh_10 = KNeighborsClassifier(n_neighbors=10)
# # knn_fit_10 = neigh_10.fit(X_train, y_train)
# # knn_pred_10 = knn_fit_10.predict_proba(X_test)
# #
# # knn_pred_m_10 = pd.DataFrame(knn_pred_10)
# # knn_pred_m_10['real'] = y_test.values
# #
# # knn_res_y_10 = np.zeros(len(y_test))
# # knn_res_y_10[knn_pred_10[:,1] >= 0.01] = 1
# # knn_res_y_10 = pd.DataFrame(knn_res_y_10)
# #
# # conf_mat_10 = confusion_matrix(y_test, knn_res_y_10.values)
# # precision_10 = precision_score(y_test, knn_res_y_10.values)
# # recall_10 = recall_score(y_test, knn_res_y_10.values)
#
# # neigh_5 = KNeighborsClassifier(n_neighbors=5)
# # knn_fit_5 = neigh_5.fit(X_train, y_train)
# # knn_pred_5 = knn_fit_5.predict_proba(X_test)
# #
# # knn_pred_m_5 = pd.DataFrame(knn_pred_5)
# # knn_pred_m_5['real'] = y_test.values
# #
# # knn_res_y_5 = np.zeros(len(y_test))
# # knn_res_y_5[knn_pred_5[:,1] >= 0.005] = 1
# # knn_res_y_5 = pd.DataFrame(knn_res_y_5)
# #
# # conf_mat_5 = confusion_matrix(y_test, knn_res_y_5.values)
# # precision_5 = precision_score(y_test, knn_res_y_5.values)
# # recall_5 = recall_score(y_test, knn_res_y_5.values)
#
# rfc = RandomForestClassifier(random_state=0, min_samples_leaf = 2, n_estimators= 100).fit(X_train, y_train)
# pred_rfc = rfc.predict_proba(X_test)
# #
# clf_pred_m_rfc = pd.DataFrame(pred_rfc)
# clf_pred_m_rfc['real'] = y_test.values
# #
# clf_res_y_rfc = np.zeros(len(y_test))
# clf_res_y_rfc[pred_rfc[:,1] >= 0.006] = 1
# clf_res_y_rfc = pd.DataFrame(clf_res_y_rfc)
# #
# conf_mat_logreg_rfc = confusion_matrix(y_test, clf_res_y_rfc.values)
# precision_logreg_rfc = precision_score(y_test, clf_res_y_rfc.values)
# recall_logreg_rfc = recall_score(y_test, clf_res_y_rfc.values)
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

brf24 = BalancedRandomForestClassifier(n_estimators=400, random_state=0).fit(X_train24h_ahead, y_train24h_ahead)
pred_brf24 = brf24.predict_proba(X_test24h_ahead)

clf_pred_m_brf24 = pd.DataFrame(pred_brf24)
#
clf_res_y_brf24 = np.zeros(len(y_test24h_ahead))
clf_res_y_brf24[pred_brf24[:,1] >= 0.1975] = 1
clf_res_y_brf24 = pd.DataFrame(clf_res_y_brf24)
#
conf_mat_logreg_brf24 = confusion_matrix(y_test24h_ahead, clf_res_y_brf24.values)
precision_logreg_brf24 = precision_score(y_test24h_ahead, clf_res_y_brf24.values)
recall_logreg_brf24 = recall_score(y_test24h_ahead, clf_res_y_brf24.values)
#prf_brfc = precision_recall_fscore_support(y_test, clf_res_y_brf.values)

f1_24 = (2 * precision_logreg_brf24 * recall_logreg_brf24) / (precision_logreg_brf24 + recall_logreg_brf24)
from numpy import sqrt
fn24 = conf_mat_logreg_brf24[1][0]
tp24 = conf_mat_logreg_brf24[1][1]
tn24 = conf_mat_logreg_brf24[0][0]
fp24 = conf_mat_logreg_brf24[0][1]
tpr24 = tp24 / (tp24 + fn24)
tnr24 = tn24 / (tn24 + fp24)
gmean24 = sqrt(tpr24 * tnr24)
wacc24 = 2/3 * tpr24 + 1/3 * tnr24


brf72 = BalancedRandomForestClassifier(n_estimators=400, random_state=0).fit(X_train72h_ahead, y_train72h_ahead)
pred_brf72 = brf72.predict_proba(X_test72h_ahead)

clf_pred_m_brf72 = pd.DataFrame(pred_brf72)
#
clf_res_y_brf72 = np.zeros(len(y_test72h_ahead))
clf_res_y_brf72[pred_brf72[:,1] >= 0.1975] = 1
clf_res_y_brf72 = pd.DataFrame(clf_res_y_brf72)
#
conf_mat_logreg_brf72 = confusion_matrix(y_test72h_ahead, clf_res_y_brf72.values)
precision_logreg_brf72 = precision_score(y_test72h_ahead, clf_res_y_brf72.values)
recall_logreg_brf72 = recall_score(y_test72h_ahead, clf_res_y_brf72.values)
#prf_brfc = precision_recall_fscore_support(y_test, clf_res_y_brf.values)

f1_72 = (2 * precision_logreg_brf72 * recall_logreg_brf72) / (precision_logreg_brf72 + recall_logreg_brf72)
from numpy import sqrt
fn72 = conf_mat_logreg_brf72[1][0]
tp72 = conf_mat_logreg_brf72[1][1]
tn72 = conf_mat_logreg_brf72[0][0]
fp72 = conf_mat_logreg_brf72[0][1]
tpr72 = tp72 / (tp72 + fn72)
tnr72 = tn72 / (tn72 + fp72)
gmean72 = sqrt(tpr72 * tnr72)
wacc72 = 2/3 * tpr72 + 1/3 * tnr72

brf120 = BalancedRandomForestClassifier(n_estimators=400, random_state=0).fit(X_train120h_ahead, y_train120h_ahead)
pred_brf120 = brf120.predict_proba(X_test120h_ahead)

clf_pred_m_brf120 = pd.DataFrame(pred_brf120)
#
clf_res_y_brf120 = np.zeros(len(y_test120h_ahead))
clf_res_y_brf120[pred_brf120[:,1] >= 0.1975] = 1
clf_res_y_brf120 = pd.DataFrame(clf_res_y_brf120)
#
conf_mat_logreg_brf120 = confusion_matrix(y_test120h_ahead, clf_res_y_brf120.values)
precision_logreg_brf120 = precision_score(y_test120h_ahead, clf_res_y_brf120.values)
recall_logreg_brf120 = recall_score(y_test120h_ahead, clf_res_y_brf120.values)
#prf_brfc = precision_recall_fscore_support(y_test, clf_res_y_brf.values)

f1_120 = (2 * precision_logreg_brf120 * recall_logreg_brf120) / (precision_logreg_brf120 + recall_logreg_brf120)
from numpy import sqrt
fn120 = conf_mat_logreg_brf120[1][0]
tp120 = conf_mat_logreg_brf120[1][1]
tn120 = conf_mat_logreg_brf120[0][0]
fp120 = conf_mat_logreg_brf120[0][1]
tpr120 = tp120 / (tp120 + fn120)
tnr120 = tn120 / (tn120 + fp120)
gmean120 = sqrt(tpr120 * tnr120)
wacc120 = 2/3 * tpr120 + 1/3 * tnr120


brf168 = BalancedRandomForestClassifier(n_estimators=400, random_state=0).fit(X_train168h_ahead, y_train168h_ahead)
pred_brf168 = brf168.predict_proba(X_test168h_ahead)

clf_pred_m_brf168 = pd.DataFrame(pred_brf168)
#
clf_res_y_brf168 = np.zeros(len(y_test168h_ahead))
clf_res_y_brf168[pred_brf168[:,1] >= 0.1975] = 1
clf_res_y_brf168 = pd.DataFrame(clf_res_y_brf168)
#
conf_mat_logreg_brf168 = confusion_matrix(y_test168h_ahead, clf_res_y_brf168.values)
precision_logreg_brf168 = precision_score(y_test168h_ahead, clf_res_y_brf168.values)
recall_logreg_brf168 = recall_score(y_test168h_ahead, clf_res_y_brf168.values)
#prf_brfc = precision_recall_fscore_support(y_test, clf_res_y_brf.values)

f1_168 = (2 * precision_logreg_brf168 * recall_logreg_brf168) / (precision_logreg_brf168 + recall_logreg_brf168)
from numpy import sqrt
fn168 = conf_mat_logreg_brf168[1][0]
tp168 = conf_mat_logreg_brf168[1][1]
tn168 = conf_mat_logreg_brf168[0][0]
fp168 = conf_mat_logreg_brf168[0][1]
tpr168 = tp168 / (tp168 + fn168)
tnr168 = tn168 / (tn168 + fp168)
gmean168 = sqrt(tpr168 * tnr168)
wacc168 = 2/3 * tpr168 + 1/3 * tnr168


print(1)