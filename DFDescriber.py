import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
#import geopandas as gpd
#import contextily as ctx

import config
import csv
import ETL
import Preprocessor

def MeteoVarsAggregatorForDunkelflauteEvents(capacities, meteo_var, country):

    DFDates = Preprocessor.HistDunkelflauteDetector(capacities, country)

    DFDates_df = pd.DataFrame(DFDates)
    #years = DFDates_df[0].apply(lambda x: x.year).unique()
    years = [1979, 1980, 1981, 1982]
    ind = 0
    for year in years:

        data_msl, dates_msl = ETL.MeterologyVarsReader([year], meteo_var)

        ind_meteo_var_DF_dates = pd.DataFrame(dates_msl).isin(DFDates)
        DF_Dates = pd.DataFrame(dates_msl)[ind_meteo_var_DF_dates.values == True]
        ind_list = ind_meteo_var_DF_dates[ind_meteo_var_DF_dates.values == True].index
        DF_Data = data_msl[ind_list, :, :]

        if ind == 0:
            DF_Data_all = DF_Data
            DF_Dates_all = DF_Dates
            ind = 1
        else:
            DF_Data_all = np.concatenate((DF_Data_all, DF_Data))
            DF_Dates_all = np.concatenate((DF_Dates_all, DF_Dates))

        data_reshaped = DF_Data_all.reshape(DF_Data_all.shape[0], -1).T.round(3)

        pd.DataFrame(data_reshaped).to_csv(
            config.file_path_ext_ssd + 'DF_Data_all_' + meteo_var[1] + '.csv', sep=';', encoding='latin1', index=False, header=False,
            quoting=csv.QUOTE_NONE)
        pd.DataFrame(DF_Dates_all).to_csv(
            config.file_path_ext_ssd + 'DF_Dates_all_' + meteo_var[1] + '.csv', sep=';', encoding='latin1', index=False, header=False,
            quoting=csv.QUOTE_NONE)
        print(str(year))

    DF_Data_all_mean = DF_Data_all.mean(axis= 0)

    pd.DataFrame(DF_Data_all_mean).to_csv(
        config.file_path_ext_ssd + 'DF_Data_all_mean_' + meteo_var[1] + '.csv', sep=';', encoding='latin1', index=False,
        header=False,
        quoting=csv.QUOTE_NONE)
    print(1)

    return DF_Data_all_mean


def MeteoVarsPlotter(DF_Data_all_mean,country_name, var_name, longitude, latitude):

    DF_Data_all_mean_df = pd.DataFrame(DF_Data_all_mean, columns = longitude)
    DF_Data_all_mean_df['latitude'] = latitude
    DF_Data_all_mean_df = DF_Data_all_mean_df.set_index('latitude')

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

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.heatmap(DF_Data_all_mean_df)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    plt.title(
        'Heatmap of mean msl in case of Dunkelflaute events for ' + country_name + ' with capacity threshold ' + str(
            config.Capacity_Threshold_DF))
    plt.savefig(
        'TestPlot_msl.png')
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
#
#     return