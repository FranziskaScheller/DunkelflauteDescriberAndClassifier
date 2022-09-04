import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

import config

import multiprocessing



def CFR_Aggregator(solar_pv_power_CFR, wind_power_ons_CFR, wind_power_offs_CFR):
    """

    :param solar_pv_power_CFR: Dataframe with solar and pv power capacity factor ratios on a country level
    :param wind_power_ons_CFR: Dataframe with onshore wind power capacity factor ratios on a country level
    :param wind_power_offs_CFR: Dataframe with offshore wind power capacity factor ratios on a maritime country level
    :return: solar_wind_aggr_CFR: Dataframe with solar and pv and wind (on- and offshore) power capacity factor ratios
             (aggregated) on a country level
    """
    # offshore wind maritime country's and onshore wind/ solar and pv country's are different
    # therefore add columns such that all dataframes have similar columns
    cols_no_offs_wind = wind_power_ons_CFR.columns.difference(wind_power_offs_CFR.columns)
    wind_power_offs_CFR[cols_no_offs_wind] = 0

    cols_only_offs_wind = wind_power_offs_CFR.columns.difference(wind_power_ons_CFR.columns)
    wind_power_ons_CFR[cols_only_offs_wind] = 0
    solar_pv_power_CFR[cols_only_offs_wind] = 0

    CFR_sum_solar_wind = solar_pv_power_CFR.append(wind_power_ons_CFR).append(wind_power_offs_CFR).groupby(['Date']).sum().reset_index()

    return CFR_sum_solar_wind


def MovingAveragesCalculator(CFR_data):

    CFR_rolling_window_mean = pd.DataFrame(CFR_data['Date'])

    CFR_rolling_window_mean[CFR_data.columns[1:]] = np.NaN

    length_mov_avg_calc_in_days_half = int((config.length_mov_avg_calc_in_days)/2)

    #for t in range(length_mov_avg_calc_in_days_half*24,length_mov_avg_calc_in_days_half*24 + 365*24,24):
    for t in range((length_mov_avg_calc_in_days_half * 24)-1, (length_mov_avg_calc_in_days_half * 24 + 365 * 24)):
        date_t = CFR_data['Date'].iloc[t]
        date_t_s = CFR_data['Date'].iloc[t - length_mov_avg_calc_in_days_half * 24]

        date_list = [date_t_s + timedelta(hours=x) for x in range(0,config.length_mov_avg_calc_in_days * 24)]

        l = len(date_list) - 1
        for x in range(0, l):
            for y in range(1, 2022 - 1979):
                new_date = date_list[x] + relativedelta(years=y)
                date_list.append(new_date)
                #print(new_date)

        date_list_mean = [date_t + relativedelta(years=y) for y in range(0, 2022-1979)]
# todo: check whether every day or every hour is needed
        #date_list_mean = [date_t + relativedelta(years=y) + relativedelta(hours=h) for y in range(0, 2022 - 1979) for h in range(0,23)]

        # calculate mean by ignoring nan values
        mean_l = np.nanmean(CFR_data[CFR_data['Date'].isin(date_list)].drop(columns='Date'), axis=0)

        for d in date_list_mean:
            CFR_rolling_window_mean.loc[CFR_rolling_window_mean['Date'] == d, CFR_data.columns[1:]] = mean_l.T

    date_t = CFR_rolling_window_mean['Date'].iloc[(length_mov_avg_calc_in_days_half * 24) -1]
    date_start = CFR_rolling_window_mean['Date'].iloc[0]
    #loc_date_t = CFR_data['Date'].iloc[length_mov_avg_calc_in_days_half * 24].index
    date_t_plus_one_year = date_t + relativedelta(years=1)
    date_start_plus_one_year = date_start + relativedelta(years=1)
    data_fill_up = CFR_rolling_window_mean[( date_start_plus_one_year <= CFR_data['Date']) & (CFR_data['Date'] <= date_t_plus_one_year)]
    #CFR_rolling_window_mean[CFR_data.columns[1:]].iloc[0:length_mov_avg_calc_in_days_half * 24] = data_fill_up[data_fill_up.columns[1:]]
    CFR_rolling_window_mean_old = CFR_rolling_window_mean.iloc[length_mov_avg_calc_in_days_half * 24:]
    CFR_rolling_window_mean_new = pd.DataFrame(
        CFR_rolling_window_mean['Date'].iloc[0:length_mov_avg_calc_in_days_half * 24])
    CFR_rolling_window_mean_new[data_fill_up.columns[1:]] = data_fill_up[data_fill_up.columns[1:]].values

    CFR_rolling_window_mean = CFR_rolling_window_mean_new.append(CFR_rolling_window_mean_old)
    print(1)
    return CFR_rolling_window_mean

def MovingAveragesCalculatorSolarPVHourly2(CFR_data):

    CFR_rolling_window_mean = pd.DataFrame(CFR_data['Date'])

    CFR_rolling_window_mean[CFR_data.columns[1:]] = np.NaN

    length_mov_avg_calc_in_days_half = int((config.length_mov_avg_calc_in_days)/2)

    #for t in range(length_mov_avg_calc_in_days_half*24,length_mov_avg_calc_in_days_half*24 + 365*24,24):
    for t in range((length_mov_avg_calc_in_days_half * 24)-1, (length_mov_avg_calc_in_days_half * 24 + 365 * 24)):
        date_t = CFR_data['Date'].iloc[t]
        date_t_s = CFR_data['Date'].iloc[t - length_mov_avg_calc_in_days_half * 24]

        date_list = [date_t_s + timedelta(hours=x) for x in range(0, config.length_mov_avg_calc_in_days * 24, 24)]

        l = len(date_list) - 1
        for x in range(0, l):
            for y in range(1, 2022 - 1979):
                new_date = date_list[x] + relativedelta(years=y)
                date_list.append(new_date)
                #print(new_date)

        date_list_mean = [date_t + relativedelta(years=y) for y in range(0, 2022-1979)]
# todo: check whether every day or every hour is needed
        #date_list_mean = [date_t + relativedelta(years=y) + relativedelta(hours=h) for y in range(0, 2022 - 1979) for h in range(0,23)]

        # calculate mean by ignoring nan values
        mean_l = np.nanmean(CFR_data[CFR_data['Date'].isin(date_list)].drop(columns='Date'), axis=0)

        for d in date_list_mean:
            CFR_rolling_window_mean.loc[CFR_rolling_window_mean['Date'] == d, CFR_data.columns[1:]] = mean_l.T

    date_t = CFR_rolling_window_mean['Date'].iloc[(length_mov_avg_calc_in_days_half * 24) -1]
    date_start = CFR_rolling_window_mean['Date'].iloc[0]
    #loc_date_t = CFR_data['Date'].iloc[length_mov_avg_calc_in_days_half * 24].index
    date_t_plus_one_year = date_t + relativedelta(years=1)
    date_start_plus_one_year = date_start + relativedelta(years=1)
    data_fill_up = CFR_rolling_window_mean[( date_start_plus_one_year <= CFR_data['Date']) & (CFR_data['Date'] <= date_t_plus_one_year)]
    #CFR_rolling_window_mean[CFR_data.columns[1:]].iloc[0:length_mov_avg_calc_in_days_half * 24] = data_fill_up[data_fill_up.columns[1:]]
    CFR_rolling_window_mean_old = CFR_rolling_window_mean.iloc[length_mov_avg_calc_in_days_half * 24:]
    CFR_rolling_window_mean_new = pd.DataFrame(
        CFR_rolling_window_mean['Date'].iloc[0:length_mov_avg_calc_in_days_half * 24])
    CFR_rolling_window_mean_new[data_fill_up.columns[1:]] = data_fill_up[data_fill_up.columns[1:]].values

    CFR_rolling_window_mean = CFR_rolling_window_mean_new.append(CFR_rolling_window_mean_old)

    return CFR_rolling_window_mean

def MovingAveragesCalculatorSolarPV(CFR_data):

    CFR_data_h_21_to_4 = CFR_data[(CFR_data['Date'].apply(lambda x: x.hour).isin([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]))]

    CFR_data_h_5_to_20 = CFR_data[(CFR_data['Date'].apply(lambda x: x.hour).isin([x for x in range(8, 18)]))]
    CFR_data_h_5_to_20 = CFR_data_h_5_to_20.reset_index().drop(columns = 'index')
    CFR_rolling_window_mean = pd.DataFrame(CFR_data_h_5_to_20['Date'])

    CFR_rolling_window_mean[CFR_data.columns[1:]] = np.NaN

    length_mov_avg_calc_in_days_half = int((config.length_mov_avg_calc_in_days)/2)

    #for t in range(length_mov_avg_calc_in_days_half*24,length_mov_avg_calc_in_days_half*24 + 365*24,24):
    for t in range((length_mov_avg_calc_in_days_half * 10), (length_mov_avg_calc_in_days_half * 10 + 365 * 10)):
        date_t = CFR_data_h_5_to_20['Date'].iloc[t]
        date_t_s = CFR_data_h_5_to_20['Date'].iloc[t - length_mov_avg_calc_in_days_half * 10]

        date_list = [date_t_s + timedelta(hours=x) for x in range(0,config.length_mov_avg_calc_in_days * 24)]
        date_list_df = pd.DataFrame(date_list)
        date_list_red = date_list_df[date_list_df[0].isin(CFR_data_h_5_to_20['Date'])]
        date_list_red = date_list_red.dropna().reset_index().drop(columns='index')

        l = len(date_list_red)
        ind = 0
        for x in range(0, l):
            for y in range(1, 2022 - 1979):
                new_date = date_list_red.loc[x, 0] + relativedelta(years=y)
                #new_date = date_list_red[x] + relativedelta(years=y)
                if ind == 0:
                    list_dates = [new_date]
                    ind = 1
                else:
                    list_dates.append(new_date)
            #print(1)

        date_list_red = date_list_red.append(pd.DataFrame(list_dates))
        date_list_mean = [date_t + relativedelta(years=y) for y in range(0, 2022-1979)]
# todo: check whether every day or every hour is needed
        #date_list_mean = [date_t + relativedelta(years=y) + relativedelta(hours=h) for y in range(0, 2022 - 1979) for h in range(0,23)]

        # calculate mean by ignoring nan values
        mean_l = np.nanmean(CFR_data[CFR_data['Date'].isin(date_list_red[0])].drop(columns='Date'), axis=0)

        for d in date_list_mean:
            CFR_rolling_window_mean.loc[CFR_rolling_window_mean['Date'] == d, CFR_data.columns[1:]] = mean_l.T

    CFR_rolling_window_mean.to_csv(
            'solar_pv_power_CFR_moving_avg_save_res.csv', sep=';', encoding='latin1', index=False)

    date_t = CFR_rolling_window_mean['Date'].iloc[(length_mov_avg_calc_in_days_half * 10) -1]
    date_start = CFR_rolling_window_mean['Date'].iloc[0]
    #loc_date_t = CFR_data['Date'].iloc[length_mov_avg_calc_in_days_half * 24].index
    date_t_plus_one_year = date_t + relativedelta(years=1)
    date_start_plus_one_year = date_start + relativedelta(years=1)
    #data_fill_up = CFR_rolling_window_mean[(date_start_plus_one_year <= CFR_data['Date']) & (CFR_data['Date'] <= date_t_plus_one_year)]
    data_fill_up = CFR_rolling_window_mean[(date_start_plus_one_year <= CFR_rolling_window_mean['Date']) & (
                CFR_rolling_window_mean['Date'] <= date_t_plus_one_year)]
    #CFR_rolling_window_mean[CFR_data.columns[1:]].iloc[0:length_mov_avg_calc_in_days_half * 24] = data_fill_up[data_fill_up.columns[1:]]
    CFR_rolling_window_mean_old = CFR_rolling_window_mean.iloc[length_mov_avg_calc_in_days_half * 10:]
    CFR_rolling_window_mean_new = pd.DataFrame(
        CFR_rolling_window_mean['Date'].iloc[0:length_mov_avg_calc_in_days_half * 10])
    CFR_rolling_window_mean_new[data_fill_up.columns[1:]] = data_fill_up[data_fill_up.columns[1:]].values

    CFR_rolling_window_mean = CFR_rolling_window_mean_new.append(CFR_rolling_window_mean_old)

    CFR_rolling_window_mean = CFR_rolling_window_mean.append(CFR_data_h_21_to_4)

    CFR_rolling_window_mean[CFR_rolling_window_mean.columns[1:]][(
        CFR_rolling_window_mean['Date'].apply(lambda x: x.hour).isin(
            [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]))] = 0

    CFR_rolling_window_mean.sort_values(by='Date', inplace=True)

    return CFR_rolling_window_mean

def MovingAveragesCalculatorSolarPVHourly(CFR_data):

    CFR_data_h_21_to_4 = CFR_data[(CFR_data['Date'].apply(lambda x: x.hour).isin([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]))]

    CFR_data_h_5_to_20 = CFR_data[(CFR_data['Date'].apply(lambda x: x.hour).isin([x for x in range(8, 18)]))]
    CFR_data_h_5_to_20 = CFR_data_h_5_to_20.reset_index().drop(columns = 'index')
    CFR_rolling_window_mean = pd.DataFrame(CFR_data_h_5_to_20['Date'])

    CFR_rolling_window_mean[CFR_data.columns[1:]] = np.NaN

    length_mov_avg_calc_in_days_half = int((config.length_mov_avg_calc_in_days)/2)

    #for t in range(length_mov_avg_calc_in_days_half*24,length_mov_avg_calc_in_days_half*24 + 365*24,24):
    for t in range((length_mov_avg_calc_in_days_half * 10), (length_mov_avg_calc_in_days_half * 10 + 365 * 10)):
        date_t = CFR_data_h_5_to_20['Date'].iloc[t]
        date_t_s = CFR_data_h_5_to_20['Date'].iloc[t - length_mov_avg_calc_in_days_half * 10]

        date_list = [date_t_s + timedelta(hours=x) for x in range(0,config.length_mov_avg_calc_in_days * 24, 24)]
        date_list_df = pd.DataFrame(date_list)
        date_list_red = date_list_df[date_list_df[0].isin(CFR_data_h_5_to_20['Date'])]
        date_list_red = date_list_red.dropna().reset_index().drop(columns='index')

        l = len(date_list_red)
        ind = 0
        for x in range(0, l):
            for y in range(1, 2022 - 1979):
                new_date = date_list_red.loc[x, 0] + relativedelta(years=y)
                #new_date = date_list_red[x] + relativedelta(years=y)
                if ind == 0:
                    list_dates = [new_date]
                    ind = 1
                else:
                    list_dates.append(new_date)
            #print(1)

        date_list_red = date_list_red.append(pd.DataFrame(list_dates))
        date_list_mean = [date_t + relativedelta(years=y) for y in range(0, 2022-1979)]
# todo: check whether every day or every hour is needed
        #date_list_mean = [date_t + relativedelta(years=y) + relativedelta(hours=h) for y in range(0, 2022 - 1979) for h in range(0,23)]

        # calculate mean by ignoring nan values
        mean_l = np.nanmean(CFR_data[CFR_data['Date'].isin(date_list_red[0])].drop(columns='Date'), axis=0)

        for d in date_list_mean:
            CFR_rolling_window_mean.loc[CFR_rolling_window_mean['Date'] == d, CFR_data.columns[1:]] = mean_l.T

    CFR_rolling_window_mean.to_csv(
            'solar_pv_power_CFR_moving_avg_save_res_hourly.csv', sep=';', encoding='latin1', index=False)

    date_t = CFR_rolling_window_mean['Date'].iloc[(length_mov_avg_calc_in_days_half * 10) -1]
    date_start = CFR_rolling_window_mean['Date'].iloc[0]
    #loc_date_t = CFR_data['Date'].iloc[length_mov_avg_calc_in_days_half * 24].index
    date_t_plus_one_year = date_t + relativedelta(years=1)
    date_start_plus_one_year = date_start + relativedelta(years=1)
    #data_fill_up = CFR_rolling_window_mean[(date_start_plus_one_year <= CFR_data['Date']) & (CFR_data['Date'] <= date_t_plus_one_year)]
    data_fill_up = CFR_rolling_window_mean[(date_start_plus_one_year <= CFR_rolling_window_mean['Date']) & (
                CFR_rolling_window_mean['Date'] <= date_t_plus_one_year)]
    #CFR_rolling_window_mean[CFR_data.columns[1:]].iloc[0:length_mov_avg_calc_in_days_half * 24] = data_fill_up[data_fill_up.columns[1:]]
    CFR_rolling_window_mean_old = CFR_rolling_window_mean.iloc[length_mov_avg_calc_in_days_half * 10:]
    CFR_rolling_window_mean_new = pd.DataFrame(
        CFR_rolling_window_mean['Date'].iloc[0:length_mov_avg_calc_in_days_half * 10])
    CFR_rolling_window_mean_new[data_fill_up.columns[1:]] = data_fill_up[data_fill_up.columns[1:]].values

    CFR_rolling_window_mean = CFR_rolling_window_mean_new.append(CFR_rolling_window_mean_old)

    CFR_rolling_window_mean = CFR_rolling_window_mean.append(CFR_data_h_21_to_4)

    CFR_rolling_window_mean[CFR_rolling_window_mean.columns[1:]][(
        CFR_rolling_window_mean['Date'].apply(lambda x: x.hour).isin(
            [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]))] = 0

    CFR_rolling_window_mean.sort_values(by='Date', inplace=True)

    return CFR_rolling_window_mean



def InstalledCapacityCorrector(CFR_moving_avg_df, CFR_df):
    """
    Function that takes the Dataframe containing the mean of the capacities of solar and PV, offshore and onshore wind
    and divides these values by the x- days moving averages of the capacities in order to normalize these values on the
    scale typically for that season of the year
    :param CFR_moving_avg_df:
    :param CFR_df:
    :return:
    """

    # in order to prevent from dividing by zero replace zeros by NaN
    CFR_moving_avg_df.replace(0, np.NaN)

    installed_capacity_solar_pv_power = CFR_df[CFR_df.columns[1:]] / CFR_moving_avg_df[CFR_moving_avg_df.columns[1:]].values

    installed_capacity_solar_pv_power.insert(loc=0, column='Date', value=CFR_df['Date'])

    return installed_capacity_solar_pv_power

def HistDunkelflauteDetector(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, country):

    # add columns for countries that don't have one specific energy source. Values are zero and therefore don't influence dunkelflaute classification
    cols_no_offs_wind = installed_capacity_factor_wind_power_ons.columns.difference(installed_capacity_factor_wind_power_offs.columns)
    installed_capacity_factor_wind_power_offs[cols_no_offs_wind] = 0

    cols_only_offs_wind = installed_capacity_factor_wind_power_offs.columns.difference(installed_capacity_factor_wind_power_ons.columns)
    installed_capacity_factor_wind_power_ons[cols_only_offs_wind] = 0
    installed_capacity_factor_solar_pv_power[cols_only_offs_wind] = 0

    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power[['Date', str(country)]]
    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power_country.rename(columns={str(country): str(country) + "_solar"})

    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons[['Date', str(country)]]
    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons_country.rename(columns={str(country): str(country) + "_wind_ons"})

    installed_capacity_factor_wind_power_offs_country = installed_capacity_factor_wind_power_offs[['Date', str(country)]]
    installed_capacity_factor_wind_power_offs_country = installed_capacity_factor_wind_power_offs_country.rename(columns={str(country): str(country) + "_wind_offs"})

    installed_capacity_factor_all = installed_capacity_factor_solar_pv_power.merge(installed_capacity_factor_wind_power_ons, on = 'Date', how = 'left')
    installed_capacity_factor_all = installed_capacity_factor_all.merge(installed_capacity_factor_wind_power_offs, on = 'Date', how = 'left')

    installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor_solar_pv_power_country[
        (installed_capacity_factor_solar_pv_power_country[str(country) + "_solar"] <= config.Capacity_Threshold_DF) &
        (installed_capacity_factor_solar_pv_power_country[str(country) + "_wind_ons"] <= config.Capacity_Threshold_DF) &
        (installed_capacity_factor_solar_pv_power_country[str(country) + "_wind_offs"] <= config.Capacity_Threshold_DF)]

    ind = 0
    for dates in installed_capacity_factor_solar_pv_power_country_df_candidates['Date']:
        range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=config.Min_length_DF - 1), freq='H')
        hour_after_range_period_df = dates + timedelta(hours=config.Min_length_DF - 1)

        if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                    [hour_after_range_period_df]).any())):

            if ind == 0:
                dunkelflaute_dates_country = range_period_df.values
                ind = 1
            else:
                dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)


    dunkelflaute_dates_country = np.unique(dunkelflaute_dates_country)
    np.savetxt('DunkelflauteDates_' + country + '_threshold_' + str(config.Capacity_Threshold_DF) + '.csv', dunkelflaute_dates_country , delimiter = ';'),

    return dunkelflaute_dates_country

def FrequencyCalculatorCFRBelowThreshold(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, country, threshold_list):

    cols_no_offs_wind = installed_capacity_factor_wind_power_ons.columns.difference(installed_capacity_factor_wind_power_offs.columns)
    installed_capacity_factor_wind_power_offs[cols_no_offs_wind] = 0

    cols_only_offs_wind = installed_capacity_factor_wind_power_offs.columns.difference(installed_capacity_factor_wind_power_ons.columns)
    installed_capacity_factor_wind_power_ons[cols_only_offs_wind] = 0
    installed_capacity_factor_solar_pv_power[cols_only_offs_wind] = 0

    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power[['Date', str(country)]]
    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power_country.rename(columns={str(country): str(country) + "_solar"})

    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons[['Date', str(country)]]
    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons_country.rename(columns={str(country): str(country) + "_wind_ons"})

    installed_capacity_factor_wind_power_offs_country = installed_capacity_factor_wind_power_offs[['Date', str(country)]]
    installed_capacity_factor_wind_power_offs_country = installed_capacity_factor_wind_power_offs_country.rename(columns={str(country): str(country) + "_wind_offs"})

    installed_capacity_factor_all = installed_capacity_factor_solar_pv_power_country.merge(installed_capacity_factor_wind_power_ons_country, on = 'Date', how = 'left')
    installed_capacity_factor_all = installed_capacity_factor_all.merge(installed_capacity_factor_wind_power_offs_country, on = 'Date', how = 'left')

    installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor_all[
        (installed_capacity_factor_all[str(country) + "_solar"] <= threshold_list[0]) &
        (installed_capacity_factor_all[str(country) + "_wind_ons"] <= threshold_list[1]) &
        (installed_capacity_factor_all[str(country) + "_wind_offs"] <= threshold_list[2])]

    DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])
    DF_frequencies['Total_Count'] = np.nan
    for l in config.range_lengths_DF_hist:
        counter_df = 0
        ind = 0
        for dates in installed_capacity_factor_solar_pv_power_country_df_candidates['Date']:
            range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
            hour_after_range_period_df = dates + timedelta(hours=l)
            hour_before_range_period_df = dates - timedelta(hours=1)

            if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                    [hour_after_range_period_df]).any()) & ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                    [hour_before_range_period_df]).any())):
                counter_df = counter_df + 1

                if ind == 0:
                    dunkelflaute_dates_country = range_period_df.values
                    ind = 1
                else:
                    dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)


        DF_frequencies['Total_Count'][DF_frequencies['LengthsDF'] == l] = counter_df

        DF_frequencies.to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + '_' + str(threshold_list[0]) + '_' + str(threshold_list[1]) + '_' + str(threshold_list[2]) + 'AC.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies

def FrequencyCalculatorCFRBelowSeveralThresholds(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, installed_capacity_factor_wind_power_offs, country, thresholds_list):

    cols_no_offs_wind = installed_capacity_factor_wind_power_ons.columns.difference(installed_capacity_factor_wind_power_offs.columns)
    installed_capacity_factor_wind_power_offs[cols_no_offs_wind] = 0

    cols_only_offs_wind = installed_capacity_factor_wind_power_offs.columns.difference(installed_capacity_factor_wind_power_ons.columns)
    installed_capacity_factor_wind_power_ons[cols_only_offs_wind] = 0
    installed_capacity_factor_solar_pv_power[cols_only_offs_wind] = 0

    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power[['Date', str(country)]]
    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power_country.rename(columns={str(country): str(country) + "_solar"})

    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons[['Date', str(country)]]
    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons_country.rename(columns={str(country): str(country) + "_wind_ons"})

    installed_capacity_factor_wind_power_offs_country = installed_capacity_factor_wind_power_offs[['Date', str(country)]]
    installed_capacity_factor_wind_power_offs_country = installed_capacity_factor_wind_power_offs_country.rename(columns={str(country): str(country) + "_wind_offs"})

    installed_capacity_factor_all = installed_capacity_factor_solar_pv_power.merge(installed_capacity_factor_wind_power_ons, on = 'Date', how = 'left')
    installed_capacity_factor_all = installed_capacity_factor_all.merge(installed_capacity_factor_wind_power_offs, on = 'Date', how = 'left')

    DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])

    for th in thresholds_list:
        DF_frequencies[str(th)] = np.nan

    for th in thresholds_list:
        installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor_solar_pv_power_country[
            (installed_capacity_factor_solar_pv_power_country[str(country) + "_solar"] <= th) &
            (installed_capacity_factor_solar_pv_power_country[str(country) + "_wind_ons"] <= th) &
            (installed_capacity_factor_solar_pv_power_country[str(country) + "_wind_offs"] <= th)]

        for l in config.range_lengths_DF_hist:
            counter_df = 0
            ind = 0
            for dates in installed_capacity_factor_solar_pv_power_country_df_candidates['Date']:
                range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
                hour_after_range_period_df = dates + timedelta(hours=l)
                hour_before_range_period_df = dates - timedelta(hours=1)

                if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                    ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_after_range_period_df]).any()) & ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_before_range_period_df]).any())):
                    counter_df = counter_df + 1

                    if ind == 0:
                        dunkelflaute_dates_country = range_period_df.values
                        ind = 1
                    else:
                        dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)

            DF_frequencies[str(th)][DF_frequencies['LengthsDF'] == l] = counter_df

        DF_frequencies.to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + 'several_thresholds.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies

def FrequencyCalculatorCFRBelowThresholdOneEnergyVariableSeveralThresholds(installed_capacity_factor, country, thresholds_list, var_type):

    DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])

    for th in thresholds_list:
        DF_frequencies[str(th)] = np.nan

    for th in thresholds_list:
        installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor[
            (installed_capacity_factor[str(country)] <= th)]

        for l in config.range_lengths_DF_hist:
            counter_df = 0
            ind = 0
            for dates in installed_capacity_factor_solar_pv_power_country_df_candidates['Date']:
                range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
                hour_after_range_period_df = dates + timedelta(hours=l)
                hour_before_range_period_df = dates - timedelta(hours=1)

                if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                    ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_after_range_period_df]).any()) & ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_before_range_period_df]).any())):
                    counter_df = counter_df + 1

                    if ind == 0:
                        dunkelflaute_dates_country = range_period_df.values
                        ind = 1
                    else:
                        dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)

            DF_frequencies[str(th)][DF_frequencies['LengthsDF'] == l] = counter_df

    DF_frequencies.to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + var_type + '_several_thresholds.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies

# def sumall(value):
#     return sum(range(1, value + 1))
#
# pool_obj = multiprocessing.Pool()
#
# answer = pool_obj.map(sumall,range(0,5))
# print(answer)

def FrequencyCalculatorCFRBelowThresholdPVOnshoreWind(installed_capacity_factor_solar_pv_power, installed_capacity_factor_wind_power_ons, country, threshold_list):

    # cols_no_offs_wind = installed_capacity_factor_wind_power_ons.columns.difference(installed_capacity_factor_wind_power_offs.columns)
    # installed_capacity_factor_wind_power_offs[cols_no_offs_wind] = 0
    #
    # cols_only_offs_wind = installed_capacity_factor_wind_power_offs.columns.difference(installed_capacity_factor_wind_power_ons.columns)
    # installed_capacity_factor_wind_power_ons[cols_only_offs_wind] = 0
    # installed_capacity_factor_solar_pv_power[cols_only_offs_wind] = 0

    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power[['Date', str(country)]]
    installed_capacity_factor_solar_pv_power_country = installed_capacity_factor_solar_pv_power_country.rename(columns={str(country): str(country) + "_solar"})

    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons[['Date', str(country)]]
    installed_capacity_factor_wind_power_ons_country = installed_capacity_factor_wind_power_ons_country.rename(columns={str(country): str(country) + "_wind_ons"})

    installed_capacity_factor_all = installed_capacity_factor_solar_pv_power_country.merge(installed_capacity_factor_wind_power_ons_country, on = 'Date', how = 'left')

    installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor_all[
        (installed_capacity_factor_all[str(country) + "_solar"] <= threshold_list[0]) &
        (installed_capacity_factor_all[str(country) + "_wind_ons"] <= threshold_list[1])]

    DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])
    DF_frequencies['Total_Count'] = np.nan
    ind = 0
    for l in config.range_lengths_DF_hist:
        counter_df = 0
        for dates in installed_capacity_factor_solar_pv_power_country_df_candidates['Date']:
            range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
            hour_after_range_period_df = dates + timedelta(hours=l)
            hour_before_range_period_df = dates - timedelta(hours=1)

            if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                    [hour_after_range_period_df]).any()) & ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                    [hour_before_range_period_df]).any())):
                counter_df = counter_df + 1

                if ind == 0:
                    dunkelflaute_dates_country = range_period_df.values
                    ind = 1
                else:
                    dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)


        DF_frequencies['Total_Count'][DF_frequencies['LengthsDF'] == l] = counter_df

        DF_frequencies.to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + '_' + str(threshold_list[0]) + '_' + str(threshold_list[1]) + '_' + '_PVOnshoreWind_AC.csv', sep=';', encoding='latin1', index=False)
        if counter_df != 0:
            pd.DataFrame(dunkelflaute_dates_country).to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + str(threshold_list[1]) +'_PVOnshoreWind_AC_dates.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies


def FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds(installed_capacity_factor, country, threshold, var_type):

    DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])

    DF_frequencies[str(threshold)] = np.nan


    installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor[
            (installed_capacity_factor[str(country)] <= threshold)]

    for l in config.range_lengths_DF_hist:
        counter_df = 0
        ind = 0
        for dates in installed_capacity_factor_solar_pv_power_country_df_candidates['Date']:
            range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
            hour_after_range_period_df = dates + timedelta(hours=l)
            hour_before_range_period_df = dates - timedelta(hours=1)

            if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                    ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_after_range_period_df]).any()) & ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_before_range_period_df]).any())):
                counter_df = counter_df + 1

                # if ind == 0:
                #     dunkelflaute_dates_country = range_period_df.values
                #     ind = 1
                # else:
                #     dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)

        DF_frequencies[str(threshold)][DF_frequencies['LengthsDF'] == l] = counter_df

        DF_frequencies.to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + var_type + str(threshold) +'_corrected_data_threshold.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies

def FrequencyCalculatorCFRBelowThresholdSolarPVOneThresholds(installed_capacity_factor, country, threshold, var_type):

    DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])

    DF_frequencies[str(threshold)] = np.nan

    installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor[
            (installed_capacity_factor[str(country)] <= threshold)]

    installed_capacity_factor_solar_pv_power_country_df_candidates_day = installed_capacity_factor_solar_pv_power_country_df_candidates[(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].apply(lambda x: x.hour).isin([x for x in range(8, 18)]))]

    #CFR_data_h_21_to_4 = CFR_data[(CFR_data['Date'].apply(lambda x: x.hour).isin([21, 22, 23, 0, 1, 2, 3, 4]))]

    #CFR_data_h_5_to_20 = CFR_data[(CFR_data['Date'].apply(lambda x: x.hour).isin([x for x in range(5, 21)]))]

    ind = 0
    for l in config.range_lengths_DF_hist:
        counter_df = 0
        print(l)
        for dates in installed_capacity_factor_solar_pv_power_country_df_candidates_day['Date']:
            range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
            hour_after_range_period_df = dates + timedelta(hours=l)
            hour_before_range_period_df = dates - timedelta(hours=1)

            if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                    ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_after_range_period_df]).any()) & ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_before_range_period_df]).any())):
                counter_df = counter_df + 1

                if ind == 0:
                    dunkelflaute_dates_country = range_period_df.values
                    ind = 1
                else:
                    dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)

        DF_frequencies[str(threshold)][DF_frequencies['LengthsDF'] == l] = counter_df

        DF_frequencies.to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + var_type + str(threshold) +'_threshold2.csv', sep=';', encoding='latin1', index=False)

        pd.DataFrame(dunkelflaute_dates_country).to_csv('CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(l) + str(country) + var_type + str(threshold) +'_threshold2_dates.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies


def HistDunkelflauteDetectorFrequencysAllCountries(installed_capacity_solar_pv_power):

    for country in installed_capacity_solar_pv_power.columns[1:]:

        installed_capacity_solar_pv_power_country = installed_capacity_solar_pv_power[['Date', str(country)]]
        installed_capacity_solar_pv_power_country_df_candidated = installed_capacity_solar_pv_power_country[installed_capacity_solar_pv_power_country[str(country)] <= config.Capacity_Threshold_DF]

        DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])
        DF_frequencies[str(country)] = np.nan
        for l in config.range_lengths_DF_hist:
            counter_df = 0
            ind = 0
            for dates in installed_capacity_solar_pv_power_country_df_candidated['Date']:
                range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
                hour_after_range_period_df = dates + timedelta(hours=l)
                hour_before_range_period_df = dates - timedelta(hours=1)

                if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_solar_pv_power_country_df_candidated['Date']).all() &
                    ~(installed_capacity_solar_pv_power_country_df_candidated['Date'].isin(
                        [hour_after_range_period_df]).any()) & ~(installed_capacity_solar_pv_power_country_df_candidated['Date'].isin(
                        [hour_before_range_period_df]).any())):
                    counter_df = counter_df + 1

                    if ind == 0:
                        dunkelflaute_dates_country = range_period_df.values
                        ind = 1
                    else:
                        dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)


            DF_frequencies[str(country)][DF_frequencies['LengthsDF'] == l] = counter_df

    DF_frequencies.to_csv('DF_relative_counts_per_nbr_of_hours_all_countries_' + str(datetime.now())+'_.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies

# def sumall(value):
#     return sum(range(1, value + 1))
#
# pool_obj = multiprocessing.Pool()
#
# answer = pool_obj.map(sumall,range(0,5))
# print(answer)

def FrequencyCalculatorCFRBelowThresholdOneEnergyVariableOneThresholds2(installed_capacity_factor, country, threshold, var_type):

    DF_frequencies = pd.DataFrame(list(config.range_lengths_DF_hist), columns=['LengthsDF'])

    DF_frequencies[str(threshold)] = np.nan


    installed_capacity_factor_solar_pv_power_country_df_candidates = installed_capacity_factor[
            (installed_capacity_factor[str(country)] <= threshold)]

    def df_calc():
        return

    for l in config.range_lengths_DF_hist:
        counter_df = 0
        ind = 0
        for dates in installed_capacity_factor_solar_pv_power_country_df_candidates['Date']:
            range_period_df = pd.date_range(start=dates, end=dates + timedelta(hours=l - 1), freq='H')
            hour_after_range_period_df = dates + timedelta(hours=l)
            hour_before_range_period_df = dates - timedelta(hours=1)

            if (pd.DataFrame(range_period_df)[0].isin(installed_capacity_factor_solar_pv_power_country_df_candidates['Date']).all() &
                    ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_after_range_period_df]).any()) & ~(installed_capacity_factor_solar_pv_power_country_df_candidates['Date'].isin(
                        [hour_before_range_period_df]).any())):
                counter_df = counter_df + 1

                # if ind == 0:
                #     dunkelflaute_dates_country = range_period_df.values
                #     ind = 1
                # else:
                #     dunkelflaute_dates_country = np.append(dunkelflaute_dates_country, range_period_df)

        DF_frequencies[str(threshold)][DF_frequencies['LengthsDF'] == l] = counter_df

        DF_frequencies.to_csv(config.file_path_ext_ssd + 'CFR_below_threshold_for_x_hrs_relative_counts_per_nbr_of_hours_' + str(country) + var_type + str(threshold) +'_threshold.csv', sep=';', encoding='latin1', index=False)

    return DF_frequencies

def HistPlotterDunkelflauteEvents(dunkelflaute_freq_country_i, country_name):

    fig, ax = plt.subplots()
    sns.histplot(data=dunkelflaute_freq_country_i, x='LengthsDF', y = dunkelflaute_freq_country_i.columns[1],
                 bins=len(dunkelflaute_freq_country_i['LengthsDF']))
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Hours')
    plt.title('Histogram of events for ' + country_name + ' where the adjusted capacity factor ratios for solar '
                    'fall below the capacity threshold ' + str(0.7) + 'for exactly x hours in a row')
    #plt.title('Histogram of events for ' + country_name + ' where the adjusted capacity factor ratios for solar and wind'
    #                ' (on- and offshore) fall below the capacity threshold ' + str(0.7) + 'for exactly x hours in a row')

    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.savefig('HistogramOfDunkelflauteEventsFor' + country_name + 'threshold ' + str(config.Capacity_Threshold_DF) + '_'+ str(datetime.today()) + '.png')
    plt.show()

def HistPlotterDunkelflauteEventsSeveralThresholdsOneCountry(dunkelflaute_freq_country_i_sevel_thresholds, country_name, var_name, threshold):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    sns.histplot(data=dunkelflaute_freq_country_i_sevel_thresholds, ax = ax1, x='LengthsDF', weights=str(dunkelflaute_freq_country_i_sevel_thresholds.columns[1]), kde=True,
                 bins=len(dunkelflaute_freq_country_i_sevel_thresholds['LengthsDF']), discrete=True)
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Hours')
    ax1.set_title('Historgram for ' + str(dunkelflaute_freq_country_i_sevel_thresholds.columns[1]))
    sns.histplot(data=dunkelflaute_freq_country_i_sevel_thresholds, ax = ax2, x='LengthsDF', weights=str(dunkelflaute_freq_country_i_sevel_thresholds.columns[2]), kde=True,
                 bins=len(dunkelflaute_freq_country_i_sevel_thresholds['LengthsDF']), discrete=True)
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Hours')
    ax2.set_title('Historgram for ' + str(dunkelflaute_freq_country_i_sevel_thresholds.columns[2]))
    sns.histplot(data=dunkelflaute_freq_country_i_sevel_thresholds, ax = ax3, x='LengthsDF', weights=str(dunkelflaute_freq_country_i_sevel_thresholds.columns[3]), kde=True,
                 bins=len(dunkelflaute_freq_country_i_sevel_thresholds['LengthsDF']), discrete=True)
    ax3.set_ylabel('Frequency')
    ax3.set_xlabel('Hours')
    ax3.set_title('Historgram for ' + str(dunkelflaute_freq_country_i_sevel_thresholds.columns[3]))

    plt.title('Histogram of events for ' + country_name + ' where the adjusted capacity factor ratios for solar and wind'
                    ' (on- and offshore) fall below the capacity threshold ' + threshold + 'for exactly x hours in a row')
    ax1.grid(axis='y')
    ax1.set_facecolor('#d8dcd6')
    txt = " This histogram is based on the data set comprising data from 1979 to 2021 and shows the frequency of events when the capacity factor of " + str(var_name) + " fall below a certain threshold."

    plt.savefig('HistogramOfDunkelflauteEventsFor' + country_name + 'threshold ' + threshold + '_'+ str(datetime.today()) + '.png')
    plt.show()


def ReferenceDateNoneDFFinder(dunkelflaute_dates):

    # Drop dates of first year (1979)
    dunkelflaute_dates_1980_to_2021 = dunkelflaute_dates['0'][dunkelflaute_dates['0'].apply(lambda x: x.year) >= 1980]

    dunkelflaute_dates_1980_to_2021_df = pd.DataFrame(dunkelflaute_dates_1980_to_2021.values,
                                                         columns=['Dates_orig'])

    dunkelflaute_dates_1980_to_2021_df['Dates_year_before'] = dunkelflaute_dates_1980_to_2021_df['Dates_orig'].apply(lambda x: x - relativedelta(years=1))

    ref_dates_in_DF_log = dunkelflaute_dates_1980_to_2021_df['Dates_year_before'].isin(dunkelflaute_dates_1980_to_2021_df['Dates_orig'])

    ref_dates_in_DF = dunkelflaute_dates_1980_to_2021_df['Dates_year_before'][ref_dates_in_DF_log == False]

    return ref_dates_in_DF

def DayBeforeDFFinder(dunkelflaute_dates):

    dunkelflaute_dates_1980_to_2021_df = pd.DataFrame(dunkelflaute_dates['0'].values,
                                                         columns=['Dates_orig'])

    dunkelflaute_dates_1980_to_2021_df['Dates_day_before'] = dunkelflaute_dates_1980_to_2021_df['Dates_orig'].apply(lambda x: x - relativedelta(days=1))

    ref_dates_in_DF_log = dunkelflaute_dates_1980_to_2021_df['Dates_day_before'].isin(dunkelflaute_dates_1980_to_2021_df['Dates_orig'])

    ref_dates_in_DF = dunkelflaute_dates_1980_to_2021_df['Dates_day_before'][ref_dates_in_DF_log == False]

    return ref_dates_in_DF