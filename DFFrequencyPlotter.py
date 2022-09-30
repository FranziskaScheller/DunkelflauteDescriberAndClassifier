import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.colors as mcolors
import seaborn as sns

def HistPlotterOneVar(data, country, variable, threshold):
    dunkelflaute_freq_country_i = data
    sns.axes_style("whitegrid")
    #sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    ax = sns.barplot(x = 'LengthsDF', y = 'Total_Count', data = dunkelflaute_freq_country_i, color = 'teal')
    #ax.get_xticklabels()[1::10]
    #for label in ax.xaxis.get_ticklabels()[::10]:
    #    label.set_visible(False)
    n = 5  # Keeps every 7th label
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    #n = 4  # Keeps every 7th label
    #[l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]
    #plt.bar(dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[0]],
    #        dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[1]])
    ax.set_ylabel('Number of Dunkelflaute Events', fontsize=13)
    ax.set_xlabel('Dunkelflaute Event Duration in Hours', fontsize=13)
    #plt.title('Frequency of dunkelflaute events in ' + country + ' where ' + variable +' falls below threshold ' + threshold + ' x hours in a row',
    #          fontsize=15)
    #plt.title('Frequency of Dunkelflaute events in ' + country + ' where PV and onshore wind fall below a threshold of ' + threshold + ' x hours in a row',
    #          fontsize=14)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    #ax.set_ylim(bottom=0, top=14)
    ax.set_facecolor('#d8dcd6')
    plt.savefig(
        'DF_Events_Plots/HistogramOfDunkelflauteEventsFor' + country + '_' + variable + '_threshold_' + threshold + str(len(dunkelflaute_freq_country_i)) + 'PVOnSh.png')
    plt.show()

def HistPlotterOneVar3Thresholds(data1, data2, data3, country, variable, thresholds_str):

    dunkelflaute_freq_country_i = data1.merge(data2, on='LengthsDF',
                                                                       how='left')
    dunkelflaute_freq_country_i = dunkelflaute_freq_country_i.merge(data3, on='LengthsDF',
                                                                    how='left')
    dunkelflaute_freq_country_i = dunkelflaute_freq_country_i.iloc[24:]

    fig, ax = plt.subplots(figsize=(20, 8), dpi=180)
    plt.bar(dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[0]] - 0.25,
            dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[1]], width=0.25, label='Threshold 0.2',
            edgecolor='black')
    plt.bar(dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[0]],
            dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[2]], width=0.25, label='Threshold 0.3',
            edgecolor='black')
    plt.bar(dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[0]] + 0.25,
            dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[3]], width=0.25, label='Threshold 0.5',
            edgecolor='black')
    ax.legend()
    ax.set_ylabel('Frequency (of CF <= threshold for x hours)', fontsize=10)
    ax.set_xlabel('Hours', fontsize=10)
    plt.title('Frequency of events in ' + country + ' where adjusted ' + variable + ' CF '
                                                         'fall below thresholds ' + thresholds_str + ' x hours in a row',
              fontsize=14)
    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.savefig(
        'DF_Events_Plots/HistogramOfDunkelflauteEventsFor' + country + '_' + variable + '_threshold_' + thresholds_str + '.png')
    plt.show()


def HistPlotterDF3Thresholds(data1, data2, data3, country, thresholds_str):

    dunkelflaute_freq_country_i = data1.merge(data2, on='LengthsDF', how='left')
    dunkelflaute_freq_country_i = dunkelflaute_freq_country_i.merge(data3, on='LengthsDF', how='left')
    dunkelflaute_freq_country_i = dunkelflaute_freq_country_i.iloc[24:]

    fig, ax = plt.subplots(figsize=(20, 8), dpi=180)
    plt.bar(dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[0]] - 0.25,
            dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[1]], width=0.25, label='Threshold 0.3',
            edgecolor='black')
    plt.bar(dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[0]],
            dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[2]], width=0.25, label='Threshold 0.5',
            edgecolor='black')
    plt.bar(dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[0]] + 0.25,
            dunkelflaute_freq_country_i[dunkelflaute_freq_country_i.columns[3]], width=0.25, label='Threshold 0.7',
            edgecolor='black')
    ax.legend()
    ax.set_ylabel('Frequency (of all CFs <= threshold for exactly x hours)', fontsize=10)
    ax.set_xlabel('Hours', fontsize=10)
    plt.title('Frequency of events in ' + country + ' where adjusted solar/PV, onshore and offshore wind CFs '
                                                         'fall below thresholds ' + thresholds_str + ' x hours in a row',
              fontsize=15)
    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.savefig(
        'DF_Events_Plots/HistogramOfDunkelflauteEventsFor' + country + 'several_thresholds.png')
    plt.show()


def DFHoursPerYearOneCountry(df_th1, df_th2, df_th3,country, list_thresholds):
    df_th1['TotalHrs'] = df_th1['LengthsDF'] * df_th1['Total_Count']
    df_th2['TotalHrs'] = df_th2['LengthsDF'] * df_th2['Total_Count']
    df_th3['TotalHrs'] = df_th3['LengthsDF'] * df_th3['Total_Count']

    df_HoursPerYearCountryi = pd.DataFrame(np.array(
        [((df_th1.iloc[24:]['TotalHrs'].sum())/42), ((df_th2.iloc[24:]['TotalHrs'].sum())/42),
         ((df_th3.iloc[24:]['TotalHrs'].sum())/42)]) , columns=country)
    df_HoursPerYearCountryi['Thresholds'] = list_thresholds

    return df_HoursPerYearCountryi

def DFHoursPerYear(df_HoursPerYearCountryi1, df_HoursPerYearCountryi2, df_HoursPerYearCountryi3):

    df_HoursPerYear = df_HoursPerYearCountryi1.merge(df_HoursPerYearCountryi2, on = 'Thresholds')
    df_HoursPerYear = df_HoursPerYear.merge(df_HoursPerYearCountryi3, on = 'Thresholds')
    return df_HoursPerYear

df_DE_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DE_0.5_0.5__PVOnshoreWind_AC.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_DE_05['product'] = df_DE_05['LengthsDF'] * df_DE_05['Total_Count']
df_DE_05_sum = df_DE_05['product'].sum()
df_DE_05_avg_sum_per_year = np.round(df_DE_05_sum/42, 3)
df_DE_05_nbr = df_DE_05['Total_Count'].sum()
df_DE_05_avg_nbr = np.round(df_DE_05['Total_Count'].sum()/42, 3)
#HistPlotterOneVar(df_DE_05, 'Germany', 'Dunkelflaute Events', '0.5')
#

df_FR_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_FR_0.5_0.5__PVOnshoreWind_AC.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_FR_05['product'] = df_FR_05['LengthsDF'] * df_FR_05['Total_Count']
df_FR_05_sum = df_FR_05['product'].sum()
df_FR_05_avg_sum_per_year = np.round(df_FR_05_sum/42, 3)
df_FR_05_nbr = df_FR_05['Total_Count'].sum()
df_FR_05_avg_nbr = np.round(df_FR_05['Total_Count'].sum()/42, 3)
#HistPlotterOneVar(df_FR_05, 'France', 'Dunkelflaute Events', '0.5')
#
df_NL_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_NL_0.5_0.5__PVOnshoreWind_AC.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_NL_05['product'] = df_NL_05['LengthsDF'] * df_NL_05['Total_Count']
df_NL_05_sum = df_NL_05['product'].sum()
df_NL_05_avg_sum_per_year = np.round(df_NL_05_sum/42, 3)
df_NL_05_nbr = df_NL_05['Total_Count'].sum()
df_NL_05_avg_nbr = np.round(df_NL_05['Total_Count'].sum()/42, 3)
#HistPlotterOneVar(df_NL_05, 'Netherlands', 'Dunkelflaute Events', '0.5')
#
df_PL_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_PL_0.5_0.5__PVOnshoreWind_AC.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_PL_05['product'] = df_PL_05['LengthsDF'] * df_PL_05['Total_Count']
df_PL_05_sum = df_PL_05['product'].sum()
df_PL_05_avg_sum_per_year = np.round(df_PL_05_sum/42, 3)
df_PL_05_nbr = df_PL_05['Total_Count'].sum()
df_PL_05_avg_nbr = np.round(df_PL_05['Total_Count'].sum()/42, 3)
#HistPlotterOneVar(df_PL_05, 'Poland', 'Dunkelflaute Events', '0.5')



dunkelflaute_freq_country_i_windows05 = pd.read_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_onshore0.5_corrected_data_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
HistPlotterOneVar(dunkelflaute_freq_country_i_windows05, 'DE', 'Onshore Wind', '0.5')

dunkelflaute_freq_country_i_windows05 = pd.read_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_offshore0.5_corrected_data_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
HistPlotterOneVar(dunkelflaute_freq_country_i_windows05, 'DE', 'Offshore Wind', '0.5')
print(1)
# Germany
#df_DE_03 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DE_0.3_0.3_0.3.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_DE_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DE_0.5_0.5__PVOnshoreWind_AC.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
HistPlotterOneVar(df_DE_05, 'DE', 'Dunkelflaute Events', '0.5')

df_DE_07 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DE_0.7_0.7_0.7.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)

HistPlotterDF3Thresholds(df_DE_03, df_DE_05, df_DE_07, 'DE', '0.3, 0.5, 0.7')
# France
df_FR_03 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_FR_0.3_0.3_0.3.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_FR_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_FR_0.5_0.5_0.5.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_FR_07 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_FR_0.7_0.7_0.7.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
#
HistPlotterDF3Thresholds(df_FR_03, df_FR_05, df_FR_07, 'FR', '0.3, 0.5, 0.7')

# Poland
df_PL_03 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_PL_0.3_0.3_0.3.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_PL_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_PL_0.5_0.5_0.5.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_PL_07 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_PL_0.7_0.7_0.7.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
#
HistPlotterDF3Thresholds(df_PL_03, df_PL_05, df_PL_07, 'PL', '0.3, 0.5, 0.7')

df_HoursPerYearDE = DFHoursPerYearOneCountry(df_DE_03, df_DE_05, df_DE_07, ['DE'], ['0.3', '0.5', '0.7'])
df_HoursPerYearFR = DFHoursPerYearOneCountry(df_FR_03, df_FR_05, df_FR_07, ['FR'], ['0.3', '0.5', '0.7'])
df_HoursPerYearPL = DFHoursPerYearOneCountry(df_PL_03, df_PL_05, df_PL_07, ['PL'], ['0.3', '0.5', '0.7'])

dfHoursPerYear = DFHoursPerYear(df_HoursPerYearDE, df_HoursPerYearFR, df_HoursPerYearPL)


# Spain
df_ES_03 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_ES_0.3_0.3_0.3.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_ES_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_ES_0.5_0.5_0.5.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
df_ES_07 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_ES_0.7_0.7_0.7.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
#
HistPlotterDF3Thresholds(df_ES_03, df_ES_05, df_ES_07, 'ES', '0.3, 0.5, 0.7')
print(1)
# dunkelflaute_freq_country_i = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.2_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i.iloc[24:], 'DE', 'solar PV', '0.2')
#
# dunkelflaute_freq_country_i = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.3_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i.iloc[24:], 'DE', 'solar PV', '0.3')
#
# dunkelflaute_freq_country_i = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.5_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i.iloc[24:], 'DE', 'solar PV', '0.5')
#
# dunkelflaute_freq_country_i = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.2_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i.iloc[0:24], 'DE', 'solar PV', '0.2')
#
# dunkelflaute_freq_country_i = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.3_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i.iloc[0:24], 'DE', 'solar PV', '0.3')
#
# dunkelflaute_freq_country_i = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.5_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i.iloc[0:24], 'DE', 'solar PV', '0.5')

# dunkelflaute_freq_country_i_02 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.2_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# dunkelflaute_freq_country_i_03 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.3_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# dunkelflaute_freq_country_i_05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEsolar_pv0.5_threshold2.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
#
# HistPlotterOneVar3Thresholds(dunkelflaute_freq_country_i_02, dunkelflaute_freq_country_i_03, dunkelflaute_freq_country_i_05, 'DE', 'Solar PV', '0.2_0.3_0.5')
#
#
# dunkelflaute_freq_country_i_windons02 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_onshore0.2_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i_windons02.iloc[24:], 'DE', 'Onshore Wind', '0.2')
# HistPlotterOneVar(dunkelflaute_freq_country_i_windons02.iloc[0:24], 'DE', 'Onshore Wind', '0.2')
#
# dunkelflaute_freq_country_i_windons03 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_onshore0.3_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i_windons03.iloc[24:], 'DE', 'Onshore Wind', '0.3')
# HistPlotterOneVar(dunkelflaute_freq_country_i_windons03.iloc[0:24], 'DE', 'Onshore Wind', '0.3')
#
# dunkelflaute_freq_country_i_windons05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_onshore0.5_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
# HistPlotterOneVar(dunkelflaute_freq_country_i_windons05.iloc[24:], 'DE', 'Onshore Wind', '0.5')
# HistPlotterOneVar(dunkelflaute_freq_country_i_windons05.iloc[0:24], 'DE', 'Onshore Wind', '0.5')
#
# HistPlotterOneVar3Thresholds(dunkelflaute_freq_country_i_windons02, dunkelflaute_freq_country_i_windons03, dunkelflaute_freq_country_i_windons05, 'DE', 'Wind Onshore', '0.2_0.3_0.5')

dunkelflaute_freq_country_i_windoffs02 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_offshore0.2_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
#HistPlotterOneVar(dunkelflaute_freq_country_i_windoffs02.iloc[24:], 'DE', 'Offshore Wind', '0.2')
#HistPlotterOneVar(dunkelflaute_freq_country_i_windoffs02.iloc[0:24], 'DE', 'Offshore Wind', '0.2')

dunkelflaute_freq_country_i_windoffs03 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_offshore0.3_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
#HistPlotterOneVar(dunkelflaute_freq_country_i_windoffs03.iloc[24:], 'DE', 'Offshore Wind', '0.3')
#HistPlotterOneVar(dunkelflaute_freq_country_i_windoffs03.iloc[0:24], 'DE', 'Offshore Wind', '0.3')

dunkelflaute_freq_country_i_windoffs05 = pd.read_csv('CFR_frequencys/CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_DEwind_power_offshore0.5_threshold.csv', error_bad_lines=False, sep=';', encoding = 'latin1', index_col= False, low_memory=False)
HistPlotterOneVar(dunkelflaute_freq_country_i_windoffs05.iloc[24:], 'DE', 'Offshore Wind', '0.5')
HistPlotterOneVar(dunkelflaute_freq_country_i_windoffs05.iloc[0:24], 'DE', 'Offshore Wind', '0.5')

HistPlotterOneVar3Thresholds(dunkelflaute_freq_country_i_windoffs02, dunkelflaute_freq_country_i_windoffs03, dunkelflaute_freq_country_i_windoffs05, 'DE', 'Wind Offshore', '0.2_0.3_0.5')



