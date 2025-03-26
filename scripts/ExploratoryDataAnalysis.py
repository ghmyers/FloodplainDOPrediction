"""
Exploratory data analysis on the river and floodplain data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.patches as mpatches

proj_dir = os.getcwd()
os.chdir(proj_dir)
data_dir = proj_dir + r'\data'
#%%
data = pd.read_csv(data_dir + r'\river_training_data.csv', index_col=[0])
data.index = pd.to_datetime(data.index).date

#%% Keep selected features
data = data.dropna(subset=['o'])
with open(data_dir + r'\vars_to_keep_FS.txt', "r") as file:
    vars_to_keep = [line.strip() for line in file]
data = data[vars_to_keep]

river_site_dfs = {}
data['site'] = data.groupby(['lat', 'lon']).ngroup()
sites = data['site'].unique()
for site in sites:
    temp_df = data[data['site'] == site]
    if len(temp_df) >= 365: # only keep sites with a year or more worth of data
        temp_df = temp_df.drop('site', axis=1)
        river_site_dfs[site] = temp_df
    else:
        data = data[data['site'] != site]
        print("not enough data")
    river_site_dfs[site] = temp_df
    
data.drop('site', axis=1, inplace=True)
#%% Create site dataframes
site_dfs = []
data['site'] = data.groupby(['lat', 'lon']).ngroup()
sites = data['site'].unique()

for site in sites:
    temp_df = data[data['site'] == site]
    site_dfs.append(temp_df)
    
#%% Plot distribution of gauges across US
usa_outline = gpd.read_file(data_dir + "/geospatial/cb_2018_us_nation_5m.shp")

# get unique gauge latitudes and longitudes and mean DO
grouped_df = data.groupby(['lat', 'lon']).agg({'o': ['mean', 'count']}).reset_index()
grouped_df.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in grouped_df.columns.values]
grouped_df = grouped_df.rename(columns={'lat_': 'lat', 'lon_': 'lon', 'o_mean': 'o', 'o_count': 'count'})

geometry = [Point(xy) for xy in zip(grouped_df['lon'], grouped_df['lat'])]
gdf = gpd.GeoDataFrame(grouped_df, geometry=geometry)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
usa_outline.plot(ax=ax, color='whitesmoke', edgecolor='black')

max_count = gdf['count'].max()
min_count = gdf['count'].min()
marker_size = (gdf['count'] - min_count) / (max_count - min_count) * 20 + 1  # Scaling factor for marker size

# Plot points color-coded by the mean of the column 'o'
gdf.plot(ax=ax, markersize=marker_size, column='o', cmap='RdBu', legend=True)

cbar = ax.get_figure().get_axes()[1]  # Get the colorbar axis
cbar.set_position([0.68, 0.1, 0.012, 0.8])  # Adjust the position and size of the colorbar
cbar.set_ylabel('Mean DO (mg/L)', rotation=270, labelpad=8)

ax.set_xlim(-145, -45)  
ax.set_ylim(25, 50)     

# Set aspect ratio
ax.set_aspect(1.3)

# Remove bounding box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Size marker based on number of observations
sizes = [1, 5, 10, 15, 20]
markers = []
for size in sizes:
    markers.append(plt.scatter([], [], s=size, edgecolor='w', facecolor='gray', alpha=0.6))
labels = [f'{int(min_count + (size - 1) * (max_count - min_count) / 19):d}' for size in sizes]

legend_markers = ax.legend(markers, labels, title='Number of \nobservations', loc='lower left', frameon=True, fontsize='small', title_fontsize='small')
ax.add_artist(legend_markers)

plt.title('480 Training Gages', fontsize=16)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.show()

#%%
OC2 = pd.read_csv(proj_dir + r"/data/OC2.csv", index_col=[0])
OC2.index = pd.to_datetime(OC2.index).date
OC2 = OC2.asfreq('D')
LF1 = pd.read_csv(proj_dir + r"/data/LF1.csv", index_col=[0])
LF1.index = pd.to_datetime(LF1.index).date
LF1 = LF1.asfreq('D')
LF2 = pd.read_csv(proj_dir + r"/data/LF2.csv", index_col=[0])
LF2.index = pd.to_datetime(LF2.index).date
LF2 = LF2.asfreq('D')
OC4 = pd.read_csv(proj_dir + r"/data/OC4.csv", index_col=[0])
OC4.index = pd.to_datetime(OC4.index).date
OC4 = OC4.asfreq('D')
LF3 = pd.read_csv(proj_dir + r"/data/LF3.csv", index_col=[0])
LF3.index = pd.to_datetime(LF3.index).date
LF3 = LF3.asfreq('D')
OC1 = pd.read_csv(proj_dir + r"/data/OC1.csv", index_col=[0])
OC1.index = pd.to_datetime(OC1.index).date
OC1 = OC1.asfreq('D')
OC3 = pd.read_csv(proj_dir + r"/data/OC3.csv", index_col=[0])
OC3.index = pd.to_datetime(OC3.index).date
OC3 = OC3.asfreq('D')
fp_site_dfs = [OC2, LF1, LF2, OC4, LF3, OC1, OC3]
site_df_dict = {"OC2": OC2,
                "LF1": LF1,
                "LF2": LF2,
                "OC4": OC4, 
                "LF3": LF3,
                "OC1": OC1,
                "OC3": OC3}
# %%
# Plot DO record, discharge, and temperature
for i, df in enumerate(fp_site_dfs):
    num_plots = 3
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
    # Plot DO record
    axes[0].plot(df['o'], label=f"Site {list(site_df_dict.keys())[i]}")
    axes[0].set_title(f"DO record at {list(site_df_dict.keys())[i]}")
    axes[0].set_ylabel("DO (mg/L)")
    axes[0].tick_params(axis='x', rotation=45)

    # Plot discharge (q)
    axes[1].plot(df['q'], label=f"Site {list(site_df_dict.keys())[i]}")
    axes[1].set_title(f"Discharge record at {list(site_df_dict.keys())[i]}")
    axes[1].set_ylabel("Discharge (ft³/s)")
    axes[1].tick_params(axis='x', rotation=45)

    # Plot temperature (temp)
    axes[2].plot(df['temp'], label=f"Site {list(site_df_dict.keys())[i]}")
    axes[2].set_title(f"Water temperature record at {list(site_df_dict.keys())[i]}")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
# %%
master_fp = pd.concat(fp_site_dfs)
master_noNA = master_fp.dropna(subset=['o'])
data_noNA = data.dropna(subset=['o'])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(data_noNA['o'], bins=30)
axes[0].set_title("a) Distribution of DO in River Data")
axes[0].set_xlabel("DO (mg/L)")
axes[0].set_ylabel("Count")
axes[0].text(0, 140000, f"n={len(data_noNA)}")
axes[1].hist(master_noNA['o'], bins=30)
axes[1].set_title("b) Distribution of DO in Floodplain Data")
axes[1].set_xlabel("DO (mg/L)")
axes[1].set_ylabel("Count")
axes[1].text(0, 390, f"n={len(master_noNA)}")
plt.tight_layout()
plt.show()