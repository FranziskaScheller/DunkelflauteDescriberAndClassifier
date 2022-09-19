from geopy.geocoders import Nominatim
import netCDF4 as nc
import numpy as np
import pandas as pd
import csv
import config
import ETL
from datetime import datetime

# initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises", timeout= 10)

# Latitude & Longitude input
Latitude = "50.00"
Longitude = "8.00"

location = geolocator.reverse(Latitude + "," + Longitude)

address = location.raw['address']

# traverse the data
city = address.get('city', '')
state = address.get('state', '')
#country = address.get('country', '')

fn = '/Volumes/PortableSSD/download19790102/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S197901010000_E197901312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
ds = nc.Dataset(fn)
#
time = ds['time'][:]
longitude = ds['longitude'][:]
latitude = ds['latitude'][:]

ssrd = ds['ssrd'][:].data
# write all coordinate combinations as columns
ssrd_reshaped = ssrd.reshape(ssrd.shape[0], -1)

if config.extract_data_GWL:
    location_df = pd.DataFrame(np.repeat(latitude, len(longitude)), columns= ['latitude'])
    longitudes = np.repeat(longitude, len(latitude))
    longitudes_resh = longitudes.reshape(len(longitude), len(latitude)).T.reshape(len(longitude)* len(latitude),)
    location_df['longitude'] = longitudes_resh
    #location_df['Country'] = location_df.apply(lambda x: geolocator.reverse(x.latitude + "," + x.longitude).raw['address'].get('country', ''))
    location_df['ind'] = np.NaN

    list_middle_points = [[70.5, -21.5], [64.75, -18.5], [54.75, -4.5], [35.75, 16.5], [41.5, -12], [62.75, 6], [56.25, 3.5], [56, 40], [68, 18], [41, 10]]
    list_edge_points = []
    for j in range(0, len(list_middle_points)):
        lat_1 = list_middle_points[j][0] - 0.5
        lat_2 = list_middle_points[j][0] + 0.5
        long_1 = list_middle_points[j][1] - 0.5
        long_2 = list_middle_points[j][1] + 0.5
        list_edge_points.insert(j, [lat_1, lat_2, long_1, long_2])

    for j in range(0, len(list_middle_points)):
        ind_mean = np.where((location_df['latitude'].isin([list_middle_points[j][0]])) & (
            location_df['longitude'].isin([list_middle_points[j][1]])) == True)[0][0]

        location_df['ind'].iloc[ind_mean] = j+1

        ind_edges_1 = np.where((location_df['latitude'].isin([list_edge_points[j][0]]) & location_df['longitude'].isin(
            [list_edge_points[j][2]])) == True)[0][0]

        ind_edges_2 = np.where((location_df['latitude'].isin([list_edge_points[j][0]]) & location_df['longitude'].isin(
            [list_edge_points[j][3]])) == True)[0][0]

        ind_edges_3 = np.where((location_df['latitude'].isin([list_edge_points[j][1]]) & location_df['longitude'].isin(
            [list_edge_points[j][2]])) == True)[0][0]

        ind_edges_4 = np.where((location_df['latitude'].isin([list_edge_points[j][1]]) & location_df['longitude'].isin(
            [list_edge_points[j][3]])) == True)[0][0]

        location_df['ind'].iloc[ind_edges_1] = j + 1
        location_df['ind'].iloc[ind_edges_2] = j + 1
        location_df['ind'].iloc[ind_edges_3] = j + 1
        location_df['ind'].iloc[ind_edges_4] = j + 1

    def MeteoVarMeanCalculatorGWL(meteo_var, location_df):


        ind = 0
        ind_2 = 0
        #for year in range(1979,2022):
        #for j in range(1, len(list_middle_points) + 1):
        for j in range(10, 11):

            indices_loc_land = location_df.index[location_df['ind'] == j]
            ind = 0
            for year in range(2011, 2022):

                data_msl, dates_msl = ETL.MeterologyVarsReader([year], meteo_var)

                if meteo_var[1] == 't2m':
                    data_msl = data_msl - 273.15

                data_msl_reshaped = data_msl.reshape(data_msl.shape[0], -1).round(3)


                data_msl_reshaped_red = data_msl_reshaped[:,indices_loc_land]

                dates_msl_df = pd.DataFrame(dates_msl, columns=['Date']).reset_index().drop(columns='index')

                #DF_Data = data_msl[:, indices_loc_land]

                DF_Data_mean = data_msl_reshaped_red.mean(axis=1)
                DF_Data_std = data_msl_reshaped_red.std(axis=1)

                #DF_Data = data_msl[ind_list, :, :]

                # DF_Dates = pd.DataFrame(dates_msl)[ind_meteo_var_DF_dates.values == True]
                # ind_list = ind_meteo_var_DF_dates[ind_meteo_var_DF_dates.values == True].index
                # DF_Data = data_msl[ind_list, :, :]


                if ind == 0:
                    DF_Data_mean_all = DF_Data_mean
                    DF_Data_std_all = DF_Data_std
                    dates_msl_all = dates_msl
                    ind = 1
                else:
                    DF_Data_mean_all = np.concatenate((DF_Data_mean_all, DF_Data_mean))
                    DF_Data_std_all = np.concatenate((DF_Data_std_all, DF_Data_std))
                    dates_msl_all = np.concatenate((dates_msl_all, dates_msl))
                    print(str(year))

            if ind_2 == 0:
                df_all = pd.DataFrame(dates_msl_all, columns=['Dates'])
                df_all['mean_' + str(j)] = DF_Data_mean_all
                df_all['std_' + str(j)] = DF_Data_std_all
                ind_2 = 1
            else:
                df_all['mean_' + str(j)] = DF_Data_mean_all
                df_all['std_' + str(j)] = DF_Data_std_all
        # pd.DataFrame(data_reshaped).to_csv(
        #         config.file_path_ext_ssd + 'DF_Data_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
        #         quoting=csv.QUOTE_NONE)
        pd.DataFrame(df_all).to_csv(
                config.file_path_ext_ssd + 'features_mean_std_GWL_j_' + str(j) + meteo_var[1]  + '11to21.csv', sep=';', encoding='latin1', index=False,
                quoting=csv.QUOTE_NONE)

        print(1)

        return DF_Data_mean_all

    #test = MeteoVarMeanCalculatorGWL(['MSL_0000m', 'msl'], location_df)
    # for j in range(1, 11):
    #     features_mean_std_mslDeutschland79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_GWL_j_' + str(j) + 'msl79to90.csv', sep = ';', encoding='latin1')
    #     features_mean_std_mslDeutschland91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_GWL_j_' + str(j) + 'msl91to00.csv', sep = ';', encoding='latin1')
    #     features_mean_std_mslDeutschland01to11 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_GWL_j_' + str(j) + 'msl01to10.csv', sep = ';', encoding='latin1')
    #     features_mean_std_mslDeutschland11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_GWL_j_' + str(j) + 'msl11to21.csv', sep = ';', encoding='latin1')
    #
    #     features_mean_std_mslDeutschland = pd.concat([features_mean_std_mslDeutschland79to90, features_mean_std_mslDeutschland91to00, features_mean_std_mslDeutschland01to11, features_mean_std_mslDeutschland11to21])
    #
    #     pd.DataFrame(features_mean_std_mslDeutschland).to_csv(
    #           config.file_path_ext_ssd + 'features_mean_std_' + 'msl_j_' + str(j) + '.csv', sep=';', index = False,
    #               encoding='latin1',
    #               quoting=csv.QUOTE_NONE)

    features_mean_std_mslGWL_1 = pd.read_csv(
        config.file_path_ext_ssd + 'features_mean_std_' + 'msl_j_' + str(1) + '.csv', sep=';', encoding='latin1')

    for j in range(2, 11):
        features_mean_std_mslGWL_j = pd.read_csv(
            config.file_path_ext_ssd + 'features_mean_std_' + 'msl_j_' + str(j) + '.csv', sep=';', encoding='latin1')

        features_mean_std_mslGWL_1 = features_mean_std_mslGWL_1.merge(features_mean_std_mslGWL_j, on = 'Dates', how = 'left')

    features_mean_std_mslGWL_1 = features_mean_std_mslGWL_1.rename({'mean_1': 'mean_Greenland', 'std_1': 'std_Greenland',
                                                                    'mean_2': 'mean_Iceland', 'std_2': 'std_Iceland',
                                                                    'mean_3': 'mean_British_Isles', 'std_3': 'std_British_Isles',
                                                                    'mean_4': 'mean_Mediterranean_Sea', 'std_4': 'std_Mediterranean_Sea',
                                                                    'mean_5': 'mean_Sea_west_Iberian_Peninsula', 'std_5': 'std_Sea_west_Iberian_Peninsula',
                                                                    'mean_6': 'mean_Norwegian_Sea', 'std_6': 'std_Norwegian_Sea',
                                                                    'mean_7': 'mean_North_Sea', 'std_7': 'std_North_Sea',
                                                                    'mean_8': 'mean_Western_Russia', 'std_8': 'std_Western_Russia',
                                                                    'mean_9': 'mean_Sweden', 'std_9': 'std_Sweden',
                                                                    'mean_10': 'mean_Tyrrhenian_Sea', 'std_10': 'std_Tyrrhenian_Sea'
                                                                    }, axis = 'columns')

    pd.DataFrame(features_mean_std_mslGWL_1).to_csv(
            config.file_path_ext_ssd + 'features_mean_std_' + 'msl_all.csv', sep=';', index = False,
            encoding='latin1',
            quoting=csv.QUOTE_NONE)
    print(1)

if config.extract_data_DE_FR_PL_NL:
    location_df = pd.DataFrame(np.repeat(latitude, len(longitude)), columns= ['latitude'])
    longitudes = np.repeat(longitude, len(latitude))
    longitudes_resh = longitudes.reshape(len(longitude), len(latitude)).T.reshape(len(longitude)* len(latitude),)
    location_df['longitude'] = longitudes_resh
    #location_df['Country'] = location_df.apply(lambda x: geolocator.reverse(x.latitude + "," + x.longitude).raw['address'].get('country', ''))
    location_df['Country'] = np.zeros(len(location_df))
    location_df['ind'] = 0

    for i in range(0,len(location_df)-1):
        if ((35 <= location_df['latitude'].iloc[i] <= 60.35) & (-5 <= location_df['longitude'].iloc[i] <= 25)):
            location_df['ind'].iloc[i] = 1
        print(i)

    # pd.DataFrame(location_df).to_csv(
    #     config.file_path_ext_ssd + 'location_df.csv', sep=';',
    #     encoding='latin1', index=False,
    #     header=False,
    #     quoting=csv.QUOTE_NONE)

    location_df_red = location_df[location_df['ind'] == 1]

    if config.FeatureCreatorLocationAdder:
        for i in range(0,len(location_df_red)):
            location_type = geolocator.reverse(str(location_df_red['latitude'].iloc[i]) + "," + str(location_df_red['longitude'].iloc[i]))

            if location_type != None:
                location_df_red['Country'].iloc[i] = location_type.raw['address'].get('country', '')

            print(i)

        pd.DataFrame(location_df_red).to_csv(
            config.file_path_ext_ssd + 'location_df_red.csv', sep=';',
            index=False,
            header=False,
            quoting=csv.QUOTE_NONE)

    # geolocator.reverse(str(location_df['latitude'].iloc[0]) + "," + str(location_df['longitude'].iloc[0])).raw['address'].get('country', '')

    location_df_red = pd.read_csv(config.file_path_ext_ssd + 'location_df_red.csv', sep = ';', header=None, names = ['latitude', 'longitude', 'Country', 'ind'])
    location_df = location_df.drop(columns =  ['Country'])
    location_df_incl_countries = location_df.merge(location_df_red[['latitude', 'longitude', 'Country']], how = 'left', on = ['latitude', 'longitude'])
    location_df_incl_countries = location_df_incl_countries.drop(columns =  ['ind'])

    def MeteoVarMeanCalculator(meteo_var, country, location_df):

        indices_loc_land = location_df.index[location_df['Country'] == country]
        ind = 0
        #for year in range(1979,2022):
        for year in range(2011, 2022):

            data_msl, dates_msl = ETL.MeterologyVarsReader([year], meteo_var)

            if meteo_var[1] == 't2m':
                data_msl = data_msl - 273.15

            data_msl_reshaped = data_msl.reshape(data_msl.shape[0], -1).round(3)

            data_msl_reshaped_red = data_msl_reshaped[:,indices_loc_land]

            dates_msl_df = pd.DataFrame(dates_msl, columns=['Date']).reset_index().drop(columns='index')

            #DF_Data = data_msl[:, indices_loc_land]

            DF_Data_mean = data_msl_reshaped_red.mean(axis=1)
            DF_Data_std = data_msl_reshaped_red.std(axis=1)

            #DF_Data = data_msl[ind_list, :, :]

            # DF_Dates = pd.DataFrame(dates_msl)[ind_meteo_var_DF_dates.values == True]
            # ind_list = ind_meteo_var_DF_dates[ind_meteo_var_DF_dates.values == True].index
            # DF_Data = data_msl[ind_list, :, :]


            if ind == 0:
                DF_Data_mean_all = DF_Data_mean
                DF_Data_std_all = DF_Data_std
                dates_msl_all = dates_msl
                ind = 1
            else:
                DF_Data_mean_all = np.concatenate((DF_Data_mean_all, DF_Data_mean))
                DF_Data_std_all = np.concatenate((DF_Data_std_all, DF_Data_std))
                dates_msl_all = np.concatenate((dates_msl_all, dates_msl))
                print(str(year))

        df_all = pd.DataFrame(dates_msl_all, columns=['Dates'])
        df_all['mean'] = DF_Data_mean_all
        df_all['std'] = DF_Data_std_all
        # pd.DataFrame(data_reshaped).to_csv(
        #         config.file_path_ext_ssd + 'DF_Data_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
        #         quoting=csv.QUOTE_NONE)
        pd.DataFrame(df_all).to_csv(
                config.file_path_ext_ssd + 'features_mean_std_' + meteo_var[1] + str(country) + '11to21.csv', sep=';', encoding='latin1', index=False,
                quoting=csv.QUOTE_NONE)



        print(1)

        return DF_Data_mean_all

    # features_mean_std_mslDeutschland79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland79to90.csv', sep = ';')
    # features_mean_std_mslDeutschland91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland91to00.csv', sep = ';')
    # features_mean_std_mslDeutschland01to11 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland01to11.csv', sep = ';')
    # features_mean_std_mslDeutschland11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslDeutschland11to21.csv', sep = ';')

    #features_mean_std_mslDeutschland = pd.concat([features_mean_std_mslDeutschland79to90, features_mean_std_mslDeutschland91to00, features_mean_std_mslDeutschland01to11, features_mean_std_mslDeutschland11to21])

    # features_mean_std_mslDeutschland79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_t2mDeutschland79to90.csv', sep = ';')
    # features_mean_std_mslDeutschland91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_t2mDeutschland91to00.csv', sep = ';')
    # features_mean_std_mslDeutschland01to11 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_t2mDeutschland01to10.csv', sep = ';')
    # features_mean_std_mslDeutschland11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_t2mDeutschland11to21.csv', sep = ';')
    #
    # features_mean_std_mslDeutschland = pd.concat([features_mean_std_mslDeutschland79to90, features_mean_std_mslDeutschland91to00, features_mean_std_mslDeutschland01to11, features_mean_std_mslDeutschland11to21])
    # #
    # pd.DataFrame(features_mean_std_mslDeutschland).to_csv(
    #      config.file_path_ext_ssd + 'features_mean_std_' + 't2m' + 'Deutschland' + '.csv', sep=';',
    #      encoding='latin1', index=False,
    #      quoting=csv.QUOTE_NONE)

    features_mean_std_mslDeutschland79to90 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance79to90.csv', sep = ';', encoding='latin1')
    features_mean_std_mslDeutschland91to00 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance91to00.csv', sep = ';', encoding='latin1')
    features_mean_std_mslDeutschland01to11 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance01to10.csv', sep = ';', encoding='latin1')
    features_mean_std_mslDeutschland11to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance11to21.csv', sep = ';', encoding='latin1')

    features_mean_std_mslDeutschland = pd.concat([features_mean_std_mslDeutschland79to90, features_mean_std_mslDeutschland91to00, features_mean_std_mslDeutschland01to11, features_mean_std_mslDeutschland11to21])

    pd.DataFrame(features_mean_std_mslDeutschland).to_csv(
          config.file_path_ext_ssd + 'features_mean_std_' + 'msl' + 'France' + '.csv', sep=';', index = False,
              encoding='latin1',
              quoting=csv.QUOTE_NONE)

    msl_aggr_FR_79to21 = pd.read_csv(config.file_path_ext_ssd + 'features_mean_std_mslFrance.csv', sep=';')

    #test = MeteoVarMeanCalculator(['TA-_0002m', 't2m'], country, location_df_incl_countries)
    #test = MeteoVarMeanCalculator(['MSL_0000m', 'msl'], 'Polska', location_df_incl_countries)
    #test = MeteoVarMeanCalculator(['MSL_0000m', 'msl'], 'Nederland', location_df_incl_countries)
    #test = MeteoVarMeanCalculator(['MSL_0000m', 'msl'], 'France', location_df_incl_countries)



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

    print(1)