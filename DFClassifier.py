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

### Create MasterTable ###
msl_aggr_DE_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland.csv', sep=';')
msl_aggr_DE_79to21['Dates'] = msl_aggr_DE_79to21['Dates'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
msl_aggr_DE_79to21 = msl_aggr_DE_79to21.rename(columns = {'Dates': 'Date', 'mean': 'mslp_mean', 'std': 'mslp_std'})


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

dunkelflaute_dates_DE = pd.read_csv(
    'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str('DE') + str(
        '0.5') + '_PVOnshoreWind_AC_dates.csv')

dunkelflaute_dates_DE['DFDates'] = dunkelflaute_dates_DE['0'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
DF_indices = msl_aggr_DE_79to21['Date'].isin(dunkelflaute_dates_DE['DFDates'])
DF_indices_0_1_encoding = DF_indices.apply(lambda x: int(x))
mastertableDFclassifier = msl_aggr_DE_79to21.merge(t2m_aggr_DE_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_PL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_NL_79to21, on = 'Date', how = 'left')
mastertableDFclassifier = mastertableDFclassifier.merge(msl_aggr_FR_79to21, on = 'Date', how = 'left')
#mastertableDFclassifier['DF_Indicator_h'] = DF_indices_0_1_encoding
mastertableDFclassifier['DF_Indicator'] = DF_indices_0_1_encoding
mastertableDFclassifier['Month'] = mastertableDFclassifier['Date'].apply(lambda x: x.month)
mastertableDFclassifier['Hour'] = mastertableDFclassifier['Date'].apply(lambda x: x.hour)

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
X = mastertableDFclassifier[['mslp_mean','mslp_std', 't2m_mean', 'Month', 'mslp_daily_mean', 'mslp_daily_std', 'mslp_daily_mean_max', 'mslp_daily_std_max', 'mslp_daily_mean_min', 'mslp_daily_std_min', 'Lag_1h_mslp_mean', 'Lag_2h_mslp_mean', 'Lag_5h_mslp_mean', 'Lag_1h_mslp_std', 'Lag_2h_mslp_std' , 'Lag_5h_mslp_std', 'Lag_120h_t2m_mean', 'Lag_120h_mslp_std', 'mslp_mean_PL', 'mslp_std_PL', 'mslp_mean_NL', 'mslp_std_NL', 'mslp_mean_FR', 'mslp_std_FR', 'Lag_120h_mslp_mean_PL', 'Lag_120h_mslp_std_PL', 'Lag_120h_mslp_mean_NL', 'Lag_120h_mslp_std_NL', 'Lag_120h_mslp_mean_FR', 'Lag_120h_mslp_std_FR', 'Lag_24h_mslp_mean_PL', 'Lag_24h_mslp_mean_NL',  'Lag_24h_mslp_mean_FR']]
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

rfc = RandomForestClassifier(random_state=0).fit(X_train, y_train)
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


clf = LogisticRegression(random_state=0, max_iter=400).fit(X_train, y_train)
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

i = 0
for year_i in res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    figure(figsize=(30, 6), dpi=80)
    data_year_i = res_eval_df_probs[
        (res_eval_df_probs['DF_ind'] == 1) & (res_eval_df_probs['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax = plt.scatter(data_year_i['Date'], data_year_i['DF_pred'])
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    plt.savefig(
        'ClassificationResults_Probabilities_' + str(year_i) + '.png')
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
    plt.savefig(
        'ClassificationResults_Probabilities_all_' + str(year_i) + '.png')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()

res_eval_df_probs_rfc = pd.DataFrame(mastertableDFclassifier[mastertableDFclassifier.index.isin(X_test.index)].Date, columns = ['Date'])
res_eval_df_probs_rfc['DF_ind'] = y_test
res_eval_df_probs_rfc['DF_pred'] = pred_rfc[:,1]

i = 0
for year_i in res_eval_df_probs_rfc[res_eval_df_probs_rfc['DF_ind'] == 1].Date.apply(lambda x: x.year).unique():
    figure(figsize=(30, 6), dpi=80)
    data_year_i = res_eval_df_probs_rfc[
        (res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax = plt.scatter(data_year_i['Date'], data_year_i['DF_pred'])
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    plt.savefig(
        'ClassificationResults_Probabilities_rfc_' + str(year_i) + '.png')
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
    plt.savefig(
        'ClassificationResults_Probabilities_all_rfc_' + str(year_i) + '.png')
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
