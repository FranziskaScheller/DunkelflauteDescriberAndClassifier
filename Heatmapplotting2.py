import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
#import warnings
#warnings.filterwarnings(action=”ignore”)

def create_geodf_from_df(df: pd.DataFrame, lon_name: str = 'longitude', lat_name: str = 'latitude') -> geopandas.GeoDataFrame:
    """ Create Geopandas.GeoDataFrame from df using the longitudinal and latitudinal information in columns lon_name and lat_name
    :param df: DataFrame with geographical information in columns lon_name and lat_name
    :param lon_name: column name with longitudinal information (default 'lon')
    :param lat_name: column name with latitudinal information (default 'lat')
    :return: geopandas.GeoDataFrame including columns of df
    """
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(x=df[lon_name], y=df[lat_name]))

# def scatter_points_germany(gdf: geopandas.GeoDataFrame,
#                            col_to_plot: str,
#                            ax: [plt.Axes, None],
#                            plot_colorbar: bool = True,
#                            **kwds_plot) -> plt.Axes:
#     """ Create scatter plot of values in gdf[col_to_plot] over boundary of germany.
#     :param gdf: geopandas data frame
#     :param col_to_plot: the column of gdf to plot
#     :param ax: (optional) a plt.Axes object where to plot
#     :param plot_colorbar: whether to plot a colorbar besides the plot
#     :param kwds_plot: passed to geopandas scatter function
#     :return: plt.Axes object with plot
#     """
#     world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#     germany = world[world['name'] == 'Germany']
#     if ax is None:
#         fig, ax = plt.subplots()
#     germany.boundary.plot(edgecolor='black', ax=ax)
#     ax.set(xlabel='longitude', ylabel='latitude')
#     if plot_colorbar:
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', legend=True, cax=cax, **kwds_plot)
#     else:
#         gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)
#     return ax

def scatter_points_europe(gdf: geopandas.GeoDataFrame,
                           col_to_plot: str,
                           ax: [plt.Axes, None],
                           plot_colorbar: bool = True,
                           **kwds_plot) -> plt.Axes:
    """ Create scatter plot of values in gdf[col_to_plot] over boundary of germany.
    :param gdf: geopandas data frame
    :param col_to_plot: the column of gdf to plot
    :param ax: (optional) a plt.Axes object where to plot
    :param plot_colorbar: whether to plot a colorbar besides the plot
    :param kwds_plot: passed to geopandas scatter function
    :return: plt.Axes object with plot
    """
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    europe = world[world.continent == "Europe"]
    #europe = world[(world.continent == "Europe") & (world.name != "Russia")]

    # Create a custom polygon
    polygon = Polygon([(-22, 26.5), (45.5, 26.5), (45.5, 72.5), (-22, 72.5)])
    # poly_gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=world.crs)
    # fig, ax = plt.subplots()
    # ax = europe.plot(ax=ax)
    # poly_gdf.plot(edgecolor=”red”, ax = ax, alpha = 0.1)
    # plt.show()

    europe = geopandas.clip(europe, polygon)
    europe.plot()

    if ax is None:
        fig, ax = plt.subplots()
    europe.boundary.plot(edgecolor='black', ax=ax)
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')
    #ax.tick_params(axis='x', labelsize=20)
    #ax.tick_params(axis='y', labelsize=20)
    #if plot_colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    #aspect='1.3',
    gdf.plot(ax=ax, marker='.', column=col_to_plot, legend=True, cax=cax, **kwds_plot)
    #ax.set(title='test')
    # else:
    #     gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)
    #gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)

    return ax


DF_Data = pd.read_csv(
    'MeanClimatologicalData_DF_DE_reshaped.csv', header=0, index_col=None, sep=';')
DF_Data = create_geodf_from_df(DF_Data)

DF_Data_24bf = pd.read_csv(
    'MeanClimatologicalData_DF_DE_reshaped_24bf.csv', header=0, index_col=None, sep=';')
DF_Data_24bf = create_geodf_from_df(DF_Data_24bf)

DF_Data_ref = pd.read_csv(
    'MeanClimatologicalData_DF_DE_reshaped_ref.csv', header=0, index_col=None, sep=';')
DF_Data_ref = create_geodf_from_df(DF_Data_ref)


#world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#europe=world[world.continent==”Europe”]

# for season in seasons:
#     wind_speeds, lat, lon, time = read_nc4_file(file=nc4_file)
#     df_mean_std = pd.DataFrame({
#         'lat': lat,
#         'lon': lon,
#         'mean': np.mean(wind_speeds, axis=0),
#         'std': np.sqrt(np.var(wind_speeds, axis=0))
#     })
#     gdf_mean_std = create_geodf_from_df(df_mean_std)
gdf_loadings = create_geodf_from_df(DF_Data)

# Air pressure
fig, axes = plt.subplots(3, 1, figsize=(10, 26), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data_24bf, 'msl_DE', ax=ax, cmap='magma')
ax.set_title('Mean air pressure at mean sea level 24 hours before a Dunkelflaute [hPa]',fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[1]
scatter_points_europe(DF_Data, 'msl_DE', ax=ax, cmap='magma')
ax.set_title('Mean air pressure at mean sea level during a Dunkelflaute [hPa]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[2]
scatter_points_europe(DF_Data_ref, 'msl_DE', ax=ax, cmap='magma')
ax.set_title('Mean air pressure at mean sea level during a reference period [hPa]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('MSLP_DE.png')

# Temperature
fig, axes = plt.subplots(3, 1, figsize=(10, 26), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data_24bf, 't2m_DE', ax=ax, cmap='magma')
ax.set_title('Mean temperature at 2 meters 24 hours before a Dunkelflaute [$^\circ$C]',fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[1]
scatter_points_europe(DF_Data, 't2m_DE', ax=ax, cmap='magma')
ax.set_title('Mean temperature at 2 meters during a Dunkelflaute [$^\circ$C]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[2]
scatter_points_europe(DF_Data_ref, 't2m_DE', ax=ax, cmap='magma')
ax.set_title('Mean temperature at 2 meters a during reference period [$^\circ$C]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('T2M_DE.png')

# Solar
fig, axes = plt.subplots(3, 1, figsize=(10, 26), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data_24bf, 'ssrdDE', ax=ax, cmap='magma')
ax.set_title('Mean solar radiation at the surface 24 hours before a Dunkelflaute [W m $^{-2}$]',fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[1]
scatter_points_europe(DF_Data, 'ssrdDE', ax=ax, cmap='magma')
ax.set_title('Mean solar radiation at the surface during a Dunkelflaute [W m $^{-2}$]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[2]
scatter_points_europe(DF_Data_ref, 'ssrdDE', ax=ax, cmap='magma')
ax.set_title('Mean solar radiation at the surface during reference period [W m $^{-2}$]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('SSRD_DE.png')

# Wind 10m
fig, axes = plt.subplots(3, 1, figsize=(10, 26), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data_24bf, 'DF_Data_all_mean_ws10DE', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 10 meters 24 hours before a Dunkelflaute [$m/s$]',fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[1]
scatter_points_europe(DF_Data, 'DF_Data_all_mean_ws10DE', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 10 meters during a Dunkelflaute [$m/s$]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[2]
scatter_points_europe(DF_Data_ref, 'DF_Data_all_mean_ws10DE', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 10 meters during reference period [$m/s$]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND10_DE.png')

# Wind 100m
fig, axes = plt.subplots(3, 1, figsize=(10, 26), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data_24bf, 'var_100_metre_wind_speedDE', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 100 meters 24 hours before a Dunkelflaute [$m/s$]',fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[1]
scatter_points_europe(DF_Data, 'var_100_metre_wind_speedDE', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 100 meters during a Dunkelflaute [$m/s$]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
ax = axes[2]
scatter_points_europe(DF_Data_ref, 'var_100_metre_wind_speedDE', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 100 meters during reference period [$m/s$]', fontsize=18)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND100_DE.png')