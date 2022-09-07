from geopy.geocoders import Nominatim
import netCDF4 as nc
import numpy as np
import pandas as pd
import csv
import config
import ETL

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
country = address.get('country', '')

fn = '/Volumes/PortableSSD/download19790102/H_ERA5_ECMW_T639_GHI_0000m_Euro_025d_S197901010000_E197901312300_INS_MAP_01h_NA-_noc_org_NA_NA---_NA---_NA---.nc'
ds = nc.Dataset(fn)
#
time = ds['time'][:]
longitude = ds['longitude'][:]
latitude = ds['latitude'][:]

ssrd = ds['ssrd'][:].data
# write all coordinate combinations as columns
ssrd_reshaped = ssrd.reshape(ssrd.shape[0], -1)

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

#for i in range(0,len(location_df_red)-1):
for i in range(6000, len(location_df_red)-1):
    location_type = geolocator.reverse(str(location_df_red['latitude'].iloc[i]) + "," + str(location_df_red['longitude'].iloc[i]))

    if location_type != None:
        location_df_red['Country'].iloc[i] = location_type.raw['address'].get('country', '')

    print(i)
pd.DataFrame(location_df_red.iloc[6000:len(location_df_red)-1]).to_csv(
    config.file_path_ext_ssd + 'location_df_red3000to6000.csv', sep=';',
    index=False,
    header=False,
    quoting=csv.QUOTE_NONE)

# geolocator.reverse(str(location_df['latitude'].iloc[0]) + "," + str(location_df['longitude'].iloc[0])).raw['address'].get('country', '')
print(1)


location_df_red = pd.read_csv('location_df_red0to4000.csv')


def MeteoVarMeanCalculator(meteo_var, country, location_df):

    indices_loc_land = location_df[location_df['Country'] == country].index()

    for year in range(1979,2021):

        data_msl, dates_msl = ETL.MeterologyVarsReader([year], meteo_var)

        data_msl_reshaped = data_msl.reshape(data_msl_reshaped.shape[0], -1).round(3)

        data_msl_reshaped_red = data_msl_reshaped.loc[:,]

        dates_msl_df = pd.DataFrame(dates_msl, columns=['Date']).reset_index().drop(columns='index')

        DF_Data = data_msl[:, indices_loc_land]

        DF_Data_mean = DF_Data.mean(axis=1)
        DF_Data_std = DF_Data.std(axis=1)

        #DF_Data = data_msl[ind_list, :, :]

        # DF_Dates = pd.DataFrame(dates_msl)[ind_meteo_var_DF_dates.values == True]
        # ind_list = ind_meteo_var_DF_dates[ind_meteo_var_DF_dates.values == True].index
        # DF_Data = data_msl[ind_list, :, :]


        if ind == 0:
            DF_Data_mean_all = DF_Data_mean
            DF_Data_std_all = DF_Data_std
            ind = 1
        else:
            DF_Data_mean_all = np.concatenate((DF_Data_mean_all, DF_Data_mean))
            DF_Data_std_all = np.concatenate((DF_Data_std_all, DF_Data_std))

            print(str(year))

    # pd.DataFrame(data_reshaped).to_csv(
    #         config.file_path_ext_ssd + 'DF_Data_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
    #         quoting=csv.QUOTE_NONE)
    pd.DataFrame(DF_Data_mean_all).to_csv(
            config.file_path_ext_ssd + 'DF_Data_mean_all_' + meteo_var[1] + str(country) + '.csv', sep=';', encoding='latin1', index=False, header=False,
            quoting=csv.QUOTE_NONE)



    print(1)

    return DF_Data_mean_all