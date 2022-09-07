import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
#import geopandas as gpd
#import contextily as ctx

import config
import csv
import ETL
import Preprocessor

def MeteoVarsAggregatorForDunkelflauteEvents(dunkelflaute_dates_country_i, meteo_var, country):

    DFDates = dunkelflaute_dates_country_i
    DFDates = DFDates.rename(columns={"0": "Date"})
    DFDates['Date'] = DFDates['Date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    DFDates_df = pd.DataFrame(DFDates)
    # DFDates_df = DFDates_df.rename(columns={"0": "Date"})
    # DFDates_df['Date'] = DFDates_df['Date'].apply(
    #     lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    years = np.sort(DFDates_df['Date'].apply(lambda x: x.year).unique())
    #years = [1979, 1980, 1981, 1982]
    ind = 0
    #for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [2000, 2001, 2002, 2003, 2004,
    #              2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:

    #for year in [2018, 2019, 2020, 2021]:
    for year in years:
    #for year in [2021]:
    #for year in [1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]:


        data_msl, dates_msl = ETL.MeterologyVarsReader([year], meteo_var)
        dates_msl_df = pd.DataFrame(dates_msl, columns=['Date']).reset_index().drop(columns='index')
        ind_meteo_var_DF_dates = dates_msl_df.merge(DFDates_df, on='Date')

        DF_Dates = dates_msl_df.merge(DFDates_df, on='Date')
        ind_list = dates_msl_df[dates_msl_df['Date'].isin(DF_Dates['Date'])].index
        DF_Data = data_msl[ind_list, :, :]

        # DF_Dates = pd.DataFrame(dates_msl)[ind_meteo_var_DF_dates.values == True]
        # ind_list = ind_meteo_var_DF_dates[ind_meteo_var_DF_dates.values == True].index
        # DF_Data = data_msl[ind_list, :, :]

        if not DF_Dates.empty:

            if ind == 0:
                DF_Data_all = DF_Data
                DF_Dates_all = DF_Dates
                ind = 1
            else:
                DF_Data_all = np.concatenate((DF_Data_all, DF_Data))
                DF_Dates_all = np.concatenate((DF_Dates_all, DF_Dates))

            data_reshaped = DF_Data_all.reshape(DF_Data_all.shape[0], -1).T.round(3)

            print(str(year))

    DF_Data_all_mean = DF_Data_all.mean(axis= 0)

    pd.DataFrame(DF_Data_all_mean).to_csv(
        config.file_path_ext_ssd + 'DF_Data_all_mean_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False,
        header=False,
        quoting=csv.QUOTE_NONE)

    # pd.DataFrame(data_reshaped).to_csv(
    #         config.file_path_ext_ssd + 'DF_Data_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
    #         quoting=csv.QUOTE_NONE)
    pd.DataFrame(DF_Dates_all).to_csv(
            config.file_path_ext_ssd + 'DF_Dates_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
            quoting=csv.QUOTE_NONE)



    print(1)

    return DF_Data_all_mean


def MeteoVarsAggregatorForDunkelflauteEventsRefDates(dunkelflaute_dates_country_i, meteo_var, country):

    DFDates = dunkelflaute_dates_country_i
    DFDates = DFDates.rename(columns={"Dates_year_before": "Date"})
    DFDates['Date'] = DFDates['Date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    DFDates_df = pd.DataFrame(DFDates)
    # DFDates_df = DFDates_df.rename(columns={"0": "Date"})
    # DFDates_df['Date'] = DFDates_df['Date'].apply(
    #     lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    years = np.sort(DFDates_df['Date'].apply(lambda x: x.year).unique())
    #years = [1979, 1980, 1981, 1982]
    ind = 0
    #for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [2000, 2001, 2002, 2003, 2004,
    #              2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:

    #for year in [2018, 2019, 2020, 2021]:
    for year in years:
    #for year in [2021]:
    #for year in [1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]:


        data_msl, dates_msl = ETL.MeterologyVarsReader([year], meteo_var)
        dates_msl_df = pd.DataFrame(dates_msl, columns=['Date']).reset_index().drop(columns='index')
        ind_meteo_var_DF_dates = dates_msl_df.merge(DFDates_df, on='Date')

        DF_Dates = dates_msl_df.merge(DFDates_df, on='Date')
        ind_list = dates_msl_df[dates_msl_df['Date'].isin(DF_Dates['Date'])].index
        DF_Data = data_msl[ind_list, :, :]

        # DF_Dates = pd.DataFrame(dates_msl)[ind_meteo_var_DF_dates.values == True]
        # ind_list = ind_meteo_var_DF_dates[ind_meteo_var_DF_dates.values == True].index
        # DF_Data = data_msl[ind_list, :, :]

        if not DF_Dates.empty:

            if ind == 0:
                DF_Data_all = DF_Data
                DF_Dates_all = DF_Dates
                ind = 1
            else:
                DF_Data_all = np.concatenate((DF_Data_all, DF_Data))
                DF_Dates_all = np.concatenate((DF_Dates_all, DF_Dates))

            data_reshaped = DF_Data_all.reshape(DF_Data_all.shape[0], -1).T.round(3)

            print(str(year))

    DF_Data_all_mean = DF_Data_all.mean(axis= 0)

    pd.DataFrame(DF_Data_all_mean).to_csv(
        config.file_path_ext_ssd + 'DF_ref_Data_all_mean_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False,
        header=False,
        quoting=csv.QUOTE_NONE)

    # pd.DataFrame(data_reshaped).to_csv(
    #         config.file_path_ext_ssd + 'DF_Data_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
    #         quoting=csv.QUOTE_NONE)
    pd.DataFrame(DF_Dates_all).to_csv(
            config.file_path_ext_ssd + 'DF_ref_Dates_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
            quoting=csv.QUOTE_NONE)



    print(1)

    return DF_Data_all_mean

def MeteoVarsAggregatorForDunkelflauteEventsDayBeforeDates(dunkelflaute_dates_country_i, meteo_var, country):

    DFDates = dunkelflaute_dates_country_i
    DFDates = DFDates.rename(columns={"Dates_day_before": "Date"})
    DFDates['Date'] = DFDates['Date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    DFDates_df = pd.DataFrame(DFDates)
    # DFDates_df = DFDates_df.rename(columns={"0": "Date"})
    # DFDates_df['Date'] = DFDates_df['Date'].apply(
    #     lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    years = np.sort(DFDates_df['Date'].apply(lambda x: x.year).unique())
    #years = [1979, 1980, 1981, 1982]
    ind = 0
    #for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
    #for year in [2000, 2001, 2002, 2003, 2004,
    #              2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:

    #for year in [2018, 2019, 2020, 2021]:
    for year in years:
    #for year in [2021]:
    #for year in [1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]:


        data_msl, dates_msl = ETL.MeterologyVarsReader([year], meteo_var)
        dates_msl_df = pd.DataFrame(dates_msl, columns=['Date']).reset_index().drop(columns='index')
        ind_meteo_var_DF_dates = dates_msl_df.merge(DFDates_df, on='Date')

        DF_Dates = dates_msl_df.merge(DFDates_df, on='Date')
        ind_list = dates_msl_df[dates_msl_df['Date'].isin(DF_Dates['Date'])].index
        DF_Data = data_msl[ind_list, :, :]

        # DF_Dates = pd.DataFrame(dates_msl)[ind_meteo_var_DF_dates.values == True]
        # ind_list = ind_meteo_var_DF_dates[ind_meteo_var_DF_dates.values == True].index
        # DF_Data = data_msl[ind_list, :, :]

        if not DF_Dates.empty:

            if ind == 0:
                DF_Data_all = DF_Data
                DF_Dates_all = DF_Dates
                ind = 1
            else:
                DF_Data_all = np.concatenate((DF_Data_all, DF_Data))
                DF_Dates_all = np.concatenate((DF_Dates_all, DF_Dates))

            data_reshaped = DF_Data_all.reshape(DF_Data_all.shape[0], -1).T.round(3)

            print(str(year))

    DF_Data_all_mean = DF_Data_all.mean(axis= 0)

    pd.DataFrame(DF_Data_all_mean).to_csv(
        config.file_path_ext_ssd + 'DF_day_bf_Data_all_mean_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False,
        header=False,
        quoting=csv.QUOTE_NONE)

    # pd.DataFrame(data_reshaped).to_csv(
    #         config.file_path_ext_ssd + 'DF_Data_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
    #         quoting=csv.QUOTE_NONE)
    pd.DataFrame(DF_Dates_all).to_csv(
            config.file_path_ext_ssd + 'DF_day_bf_Dates_all_mean_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
            quoting=csv.QUOTE_NONE)



    print(1)

    return DF_Data_all_mean



def MeteoVarsPlotter(DF_Data_all_mean,country_name, var_name, longitude, latitude, label):

    DF_Data_all_mean_df = pd.DataFrame(DF_Data_all_mean)
    DF_Data_all_mean_df.columns = longitude
    DF_Data_all_mean_df.index = latitude

    #DF_Data_all_mean_df['latitude'] = latitude
    #DF_Data_all_mean_df = DF_Data_all_mean_df.set_index('latitude')

    # path_rg = config.file_path_ext_ssd + "NUTS_RG_01M_2021_3035.json"
    # gdf_rg = gpd.read_file(path_rg)
    # path_bn = config.file_path_ext_ssd + "NUTS_BN_01M_2021_3035.json"
    # gdf_bn = gpd.read_file(path_bn)
    # path_lb = config.file_path_ext_ssd + "NUTS_LB_2021_3035.json"
    # gdf_lb = gpd.read_file(path_lb)
    #
    # ax = gdf_rg.plot(figsize=(20, 15), color="gray")
    # gdf_bn.plot(figsize=(20, 15), ax=ax, color="red")
    # gdf_lb.plot(figsize=(20, 15), ax=ax, color="yellow")

    fig, ax = plt.subplots(figsize = (8,6))
    sns.heatmap(DF_Data_all_mean, cbar_kws={'label': label})
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xlabel('Longitude', fontsize=10)
    plt.title(
        'Mean of ' + str(var_name) + ' in case of Dunkelflaute events for ' + country_name + ' with threshold ' + str(
            config.Capacity_Threshold_DF), fontsize=10)
    plt.savefig(
        'Plot_mean_' + str(var_name) + '_' + str(country_name) + '.png')
    plt.show()

    print(1)
    return

def MeteoVarsPlotterRefDate(DF_Data_all_mean,country_name, var_name, longitude, latitude, label):

    DF_Data_all_mean_df = pd.DataFrame(DF_Data_all_mean)
    DF_Data_all_mean_df.columns = longitude
    DF_Data_all_mean_df.index = latitude

    fig, ax = plt.subplots(figsize = (8,6))
    sns.heatmap(DF_Data_all_mean, cbar_kws={'label': label})
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xlabel('Longitude', fontsize=10)
    plt.title(
        'Mean of ' + str(var_name) + ' in case of no Dunkelflaute events for ' + country_name, fontsize=10)
    plt.savefig(
        'Plot_mean_reference_no_DF_' + str(var_name) + '_' + str(country_name) + '.png')
    plt.show()

    return

def MeteoVarsPlotterBeforeDF(DF_Data_all_mean,country_name, var_name, longitude, latitude, label):

    DF_Data_all_mean_df = pd.DataFrame(DF_Data_all_mean)
    DF_Data_all_mean_df.columns = longitude
    DF_Data_all_mean_df.index = latitude

    fig, ax = plt.subplots(figsize = (8,6))
    sns.heatmap(DF_Data_all_mean, cbar_kws={'label': label})
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xlabel('Longitude', fontsize=10)
    plt.title(
        'Mean of ' + str(var_name) + ' 24 hours before Dunkelflaute event for ' + country_name, fontsize=10)
    plt.savefig(
        'Plot_mean_24hrs_before_DF_' + str(var_name) + '_' + str(country_name) + '.png')
    plt.show()

    return


# def MeteoVarsPlotter(dunkelflaute_date_list, data, dates, var_name, longitude, latitude):
#
#     date_index = dates[dunkelflaute_date_list]
#
#     date_index = pd.DataFrame(data, columns = 'Date').index[data['Date'].isin(dunkelflaute_date_list)]
#
#     data_DF = data.iloc[date_index]
#
#     means = np.mean(data_DF, axis=0)
#
#     means = means.reshape(len(longitude), len(latitude))
# #
#     return