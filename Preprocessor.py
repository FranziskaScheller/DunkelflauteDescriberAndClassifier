import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import config


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


def MovingAveragesCalculator(CFR_sum_solar_wind):

    solar_pv_wind_power_moving_avg = pd.DataFrame(CFR_sum_solar_wind['Date'])

    solar_pv_wind_power_moving_avg[CFR_sum_solar_wind.columns[1:]] = np.NaN

    length_mov_avg_calc_in_days_half = int((config.length_mov_avg_calc_in_days)/2)

    #for t in range(length_mov_avg_calc_in_days_half*24,length_mov_avg_calc_in_days_half*24 + 365*24,24):
    for t in range(length_mov_avg_calc_in_days_half * 24, length_mov_avg_calc_in_days_half * 24 + 365 * 24):
        date_t = CFR_sum_solar_wind['Date'].iloc[t]
        date_t_s = CFR_sum_solar_wind['Date'].iloc[t-length_mov_avg_calc_in_days_half*24]

        date_list = [date_t_s + timedelta(hours=x) for x in range(0,config.length_mov_avg_calc_in_days * 24)]

        l = len(date_list) - 1
        for x in range(0, l):
            for y in range(1, 2022 - 1979):
                new_date = date_list[x] + relativedelta(years=y)
                date_list.append(new_date)
                #print(new_date)

        date_list_mean = [date_t + relativedelta(years=y) for y in range(0,2022-1979)]
# todo: check whether every day or every hour is needed
        #date_list_mean = [date_t + relativedelta(years=y) + relativedelta(hours=h) for y in range(0, 2022 - 1979) for h in range(0,23)]

        # calculate mean by ignoring nan values
        mean_l = np.nanmean(CFR_sum_solar_wind[CFR_sum_solar_wind['Date'].isin(date_list)].drop(columns='Date'), axis=0)

        for d in date_list_mean:
            solar_pv_wind_power_moving_avg.loc[solar_pv_wind_power_moving_avg['Date'] == d, CFR_sum_solar_wind.columns[1:]] = mean_l.T

    return solar_pv_wind_power_moving_avg

def InstalledCapacityCorrector(solar_pv_wind_power_moving_avg, CFR_sum_solar_wind):
    """
    Function that takes the Dataframe containing the mean of the capacities of solar and PV, offshore and onshore wind
    and divides these values by the x- days moving averages of the capacities in order to normalize these values on the
    scale typically for that season of the year
    :param solar_pv_wind_power_moving_avg:
    :param CFR_sum_solar_wind:
    :return:
    """

    # in order to prevent from dividing by zero replace zeros by NaN
    solar_pv_wind_power_moving_avg.replace(0, np.NaN)

    installed_capacity_solar_pv_power = CFR_sum_solar_wind[CFR_sum_solar_wind.columns[1:]] / (solar_pv_wind_power_moving_avg[
        solar_pv_wind_power_moving_avg.columns[1:]].values * 3)

    installed_capacity_solar_pv_power.insert(loc=0, column='Date', value=CFR_sum_solar_wind['Date'])

    return installed_capacity_solar_pv_power

def HistDunkelflauteDetector(installed_capacity_solar_pv_power):

    for country in installed_capacity_solar_pv_power.columns[1:]:

        country = installed_capacity_solar_pv_power[['Date', str(country)]][installed_capacity_solar_pv_power[str(country)] <= config.Capacity_Threshold_DF]

    dunkelflaute_date_list = 1

    return dunkelflaute_date_list