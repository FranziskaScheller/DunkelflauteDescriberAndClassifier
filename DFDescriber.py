import pandas as pd
import numpy as np

def MeteoVarsPlotter(dunkelflaute_date_list, data, dates, var_name, longitude, latitude):

    date_index = dates[dunkelflaute_date_list]

    date_index = pd.DataFrame(data, columns = 'Date').index[data['Date'].isin(dunkelflaute_date_list)]

    data_DF = data.iloc[date_index]

    means = np.mean(data_DF, axis=0)

    means = means.reshape(len(longitude), len(latitude))

    return