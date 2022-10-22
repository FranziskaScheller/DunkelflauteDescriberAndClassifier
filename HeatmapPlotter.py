import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_geodf_from_df(df: pd.DataFrame, lon_name: str = 'longitude', lat_name: str = 'latitude') -> geopandas.GeoDataFrame:
    """ Create Geopandas.GeoDataFrame from df using the longitudinal and latitudinal information in columns lon_name and lat_name
    :param df: DataFrame with geographical information in columns lon_name and lat_name
    :param lon_name: column name with longitudinal information (default 'lon')
    :param lat_name: column name with latitudinal information (default 'lat')
    :return: geopandas.GeoDataFrame including columns of df
    """
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(x=df[lon_name], y=df[lat_name]))

def scatter_points_germany(gdf: geopandas.GeoDataFrame,
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
    germany = world[world['name'] == 'Germany']
    if ax is None:
        fig, ax = plt.subplots()
    germany.boundary.plot(edgecolor='black', ax=ax)
    ax.set(xlabel='longitude', ylabel='latitude')
    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', legend=True, cax=cax, **kwds_plot)
    else:
        gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)
    return ax

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
    world = geopandas.read_file(geopandas.datasets.get_path(‘naturalearth_lowres’))
    europe = world[world.continent ==”Europe”]

    if ax is None:
        fig, ax = plt.subplots()
    europe.boundary.plot(edgecolor='black', ax=ax)
    ax.set(xlabel='longitude', ylabel='latitude')
    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', legend=True, cax=cax, **kwds_plot)
    else:
        gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)
    return ax

# Load data

DF_Data = pd.read_csv(
    'MeanClimatologicalData_DF_DE_reshaped.csv', header=0, index_col=None, sep=';')

gdf_loadings = create_geodf_from_df(DF_Data)

world = geopandas.read_file(geopandas.datasets.get_path(‘naturalearth_lowres’))
europe=world[world.continent==”Europe”]

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

fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height / 1.3))
ax = axes[0]
scatter_points_germany(gdf_loadings, f'mean', ax=ax, cmap=BuGn)
ax.set_title('Mean [$m/s$]')


ax = axes[1]
scatter_points_germany(gdf_mean_std, f'std', ax=ax, cmap=cycler_01)
ax.set_title('Standard deviation [$m/s$]')
#fig.tight_layout()
#fig.savefig(join(exportfolder, f'mean_std_spatially_season_{season}.png'), dpi=dpi)

