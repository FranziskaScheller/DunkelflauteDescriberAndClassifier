from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib

installed_capacity_factor_solar_pv_power = pd.read_csv('installed_capacity_factor_solar_pv_power_h2.csv',
                                                       error_bad_lines=False, sep=';', encoding='latin1',
                                                       index_col=False, low_memory=False)
installed_capacity_factor_wind_power_ons = pd.read_csv('installed_capacity_factor_wind_power_ons.csv',
                                                       error_bad_lines=False, sep=';', encoding='latin1',
                                                       index_col=False, low_memory=False)

installed_capacity_factor_solar_pv_power['Date'] = installed_capacity_factor_solar_pv_power['Date'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
installed_capacity_factor_wind_power_ons['Date'] = installed_capacity_factor_wind_power_ons['Date'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power.fillna(0)
installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power.round(3)
installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons.round(3)

installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power[installed_capacity_factor_solar_pv_power['Date'].dt.year.isin(range(2007, 2022))].reset_index().drop(columns = 'index')
installed_capacity_factor_solar_pv_power = installed_capacity_factor_solar_pv_power[['Date', 'DE']]

installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons[installed_capacity_factor_wind_power_ons['Date'].dt.year.isin(range(2007, 2022))].reset_index().drop(columns = 'index')
installed_capacity_factor_wind_power_ons = installed_capacity_factor_wind_power_ons[['Date', 'DE']]

installed_capacity_factor_wind_power_ons_plus_solar = pd.DataFrame(installed_capacity_factor_wind_power_ons['Date'],  columns= ['Date'])
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.merge(installed_capacity_factor_solar_pv_power, on = 'Date', how = 'left')
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.rename({'DE': 'solar_DE'}, axis = 'columns')
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.merge(installed_capacity_factor_wind_power_ons, on = 'Date', how = 'left')
installed_capacity_factor_wind_power_ons_plus_solar = installed_capacity_factor_wind_power_ons_plus_solar.rename({'DE': 'wind_ons_DE'}, axis = 'columns')
installed_capacity_factor_wind_power_ons_plus_solar['sum'] = installed_capacity_factor_wind_power_ons_plus_solar['solar_DE'] + installed_capacity_factor_wind_power_ons_plus_solar['wind_ons_DE']

i = 0
for year_i in installed_capacity_factor_wind_power_ons.Date.apply(lambda x: x.year).unique():
    fig, ax = plt.subplots(3, figsize=(30, 6), dpi=80)
    data_wind_year_i = installed_capacity_factor_wind_power_ons[
        installed_capacity_factor_wind_power_ons['Date'].apply(lambda x: x.year).isin([year_i])]
    data_solar_year_i = installed_capacity_factor_solar_pv_power[
        installed_capacity_factor_solar_pv_power['Date'].apply(lambda x: x.year).isin([year_i])]
    data_sum_year_i = installed_capacity_factor_wind_power_ons_plus_solar[
        installed_capacity_factor_wind_power_ons_plus_solar['Date'].apply(lambda x: x.year).isin([year_i])]

    #data_year_i_df = res_eval_df_probs_rfc[(res_eval_df_probs_rfc['DF_ind'] == 1) & (res_eval_df_probs_rfc['Date'].apply(lambda x: x.year).isin([year_i]))]
    ax[0].plot(data_wind_year_i['Date'], data_wind_year_i['DE'])
    ax[1].plot(data_solar_year_i['Date'], data_solar_year_i['DE'])
    ax[2].plot(data_sum_year_i['Date'], data_sum_year_i['sum'])
    #ax = plt.scatter(data_year_i_df['Date'], data_year_i_df['DF_pred'])
    ax[0].hlines(0.5, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    ax[1].hlines(0.5, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    ax[2].hlines(1, data_solar_year_i['Date'].iloc[0], data_solar_year_i['Date'].iloc[-1], 'green')
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    plt.savefig(
        'ClassificationResults_Probabilities_all_rfc_' + str(year_i) + '.png')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #plt.plot(res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].Date, res_eval_df_probs[res_eval_df_probs['DF_ind'] == 1].DF_pred)
    i = i + 1
    plt.show()
