from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import config
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

### Create MasterTable ###
msl_aggr_DE_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland.csv', sep=';')
msl_aggr_DE_79to21['Dates'] = msl_aggr_DE_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_DE_79to21 = msl_aggr_DE_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp_mean', 'std': 'mslp_std'})

t2m_aggr_DE_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_t2mDeutschland.csv', sep=';')
t2m_aggr_DE_79to21['Dates'] = t2m_aggr_DE_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
t2m_aggr_DE_79to21 = t2m_aggr_DE_79to21.rename(columns = {'Dates': 'Date', 'mean': 't2m_mean', 'std': 't2m_std'})

dunkelflaute_dates_DE = pd.read_csv(
    'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('DE') + str(
        '0.5') + '_PVOnshoreWind_AC_dates.csv')

dunkelflaute_dates_DE['DFDates'] = dunkelflaute_dates_DE['0'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
DF_indices = msl_aggr_DE_79to21['Date'].isin(dunkelflaute_dates_DE['DFDates'])
DF_indices_0_1_encoding = DF_indices.apply(lambda x: int(x))
mastertableDFclassifier = msl_aggr_DE_79to21.merge(t2m_aggr_DE_79to21, on = 'Date', how = 'left')
mastertableDFclassifier['DF_Indicator'] = DF_indices_0_1_encoding

X = mastertableDFclassifier[['mslp_mean', 'mslp_std', 't2m_mean', 't2m_std']]
y = mastertableDFclassifier['DF_Indicator']
neigh = KNeighborsClassifier(n_neighbors=20)
knn_fit = neigh.fit(X, y)


cv_score = cross_val_score(neigh, X, y, cv=5)
print(1)
