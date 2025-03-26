"""
Feature selection on river training and floodplain site data
Harrison Myers
6/26/2024
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
#%%
proj_dir = os.getcwd()
os.chdir(proj_dir)
data_dir = proj_dir + "\\data"

train_data = pd.read_csv(data_dir + "\\river_training_data.csv", index_col=[0])
train_data.index = pd.to_datetime(train_data.index).date

#%% 
site_data = []
for f in os.listdir(data_dir):
    if f.startswith('river'):
        pass
    else:
        df = pd.read_csv(data_dir + "/" + f, index_col=[0])
        df.index = pd.to_datetime(df.index).date
        site_data.append(df)
master = pd.concat(site_data)
master.sort_index(axis=1, inplace=True)
train_data.sort_index(axis=1, inplace=True)

#%% Compute correlation matrix
corr_matrix_riv = train_data.corr()
# Create a mask to display only the upper triangle
mask_riv = np.triu(np.ones_like(corr_matrix_riv, dtype=bool))
# Set up the matplotlib figure
plt.figure(figsize=(16, 12))

# Draw the heatmap
sns.heatmap(corr_matrix_riv, mask=mask_riv, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
#%% Identify highly correlated features (r > 0.9) and store them to drop later
threshold = 0.9  # Set your threshold
cols_to_remove = set()

for i in range(len(corr_matrix_riv.columns)):
    for j in range(i):
        if abs(corr_matrix_riv.iloc[i, j]) > threshold:
            colname = corr_matrix_riv.columns[i]
            cols_to_remove.add(colname)

print(cols_to_remove)

#%% Drop variables with less than 5% 0 or NA values
# Replace NaN values with zero and convert to boolean (True for non-zero values)
boolean_df = train_data.fillna(0).astype(bool)

# Sum the boolean DataFrame along the columns to get the count of non-zero and non-NaN values
non_zero_non_nan_prop = boolean_df.sum(axis=0) / len(train_data)

# Remove columns with less than 25% non-zero, non-na observations
for i, var in enumerate(non_zero_non_nan_prop):
    if var < 0.25:
        cols_to_remove.add(list(boolean_df.columns)[i])
print(cols_to_remove)

#%%
# Drop highly correlated features
corr_feats_to_keep = ['lat',
                      'swe (kg/m2)', 
                      'tmin (degrees C)', 
                      'vp (Pa)']
for feat in corr_feats_to_keep:
    cols_to_remove.remove(feat)
train_data_reduced = train_data.drop(columns=cols_to_remove)
master_reduced = master.drop(columns=cols_to_remove)

#%% Scale the data
scaler_y = MinMaxScaler()
scaler_y.fit(np.array(master["o"]).reshape(-1, 1))

# Scale dataset
scaler = MinMaxScaler()
scaler.fit(master_reduced) 
master_sc = pd.DataFrame(scaler.transform(master_reduced.values), 
                             columns=master_reduced.columns, 
                             index=master_reduced.index)

scaler_y_riv = MinMaxScaler()
scaler_y_riv.fit(np.array(train_data_reduced['o']).reshape(-1, 1))
scaler_riv = MinMaxScaler()
scaler_riv.fit(train_data_reduced)
train_data_sc = pd.DataFrame(scaler_riv.transform(train_data_reduced.values), 
                             columns=train_data_reduced.columns, 
                             index=train_data_reduced.index)

#%% add column for site to get even train:test splits
master_sc['site'] = master_sc.groupby(['lat', 'lon']).ngroup()
train_data_sc['site'] = train_data_sc.groupby(['lat', 'lon']).ngroup()

# master_sc['site'] = (master_sc.index.to_series().diff().dt.days != 1).cumsum()
# train_data_sc['site'] = (train_data_sc.index.to_series().diff().dt.days != 1).cumsum()

# Drop NaNs for feature selection
master_sc.dropna(inplace=True)
train_data_sc.dropna(inplace=True)

def train_test_split_evenSites(df, split_pct):
    sites = df['site'].unique()
    train_splits = []
    test_splits  = []
    split_inds = []
    for site in sites:
        temp_df = df[df['site'] == site]
        split_ind = int(np.floor(split_pct*int(len(temp_df))))
        split_inds.append(split_ind)
        train_df = temp_df.iloc[:split_ind, :]
        test_df = temp_df.iloc[split_ind:, :]
        train_splits.append(train_df)
        test_splits.append(test_df)
    Train = pd.concat(train_splits)
    Test = pd.concat(test_splits)
    
    return Train, Test, split_inds

master_train, master_test, master_split_inds = train_test_split_evenSites(master_sc, 0.8)
river_train, river_test, river_split_inds = train_test_split_evenSites(train_data_sc, 0.8)

#%% Get X, y
master_train.drop('site', axis=1, inplace=True)
master_test.drop('site', axis=1, inplace=True)
river_train.drop('site', axis=1, inplace=True)
river_test.drop('site', axis=1, inplace=True)

master_y_train = master_train.pop("o")
master_y_test = master_test.pop('o')
river_y_train = river_train.pop("o")
river_y_test = river_test.pop("o")

#%% Create dictionary of variable names with descriptions, labels, and plotting colors
feat_dict = {'lat': ("Latitude", "Physical Attributes", '#7D7D7D'),
            "FUNGICIDE": ('Fungicide use', 'Land Use / Land Cover', '#77B28C'),
             "ELEV_MEAN": ("Mean elevation (m)", "Physical Attributes", '#7D7D7D'),
             "LAKEPOND": ("% Lakes or ponds", 'Land Use / Land Cover', '#77B28C'),
             "N97": ("Nitrogen from fertilizer/manure (1997)", 'Land Use / Land Cover', '#77B28C'),
             "CNPY11_BUFF100": ("% Tree canopy in 100m riparian buffer", 'Land Use / Land Cover', '#77B28C'),
             "BASIN_AREA": ("Catchment area", "Physical Attributes", '#7D7D7D'),
             "NLCD01_52": ("% Shrub/scrub", 'Land Use / Land Cover', '#77B28C'),
             "ELEV_MAX": ("Maximum catchment elevation (m)", "Physical Attributes", '#7D7D7D'),
             "dayl (s)": ("Day length (s)", 'Meteorological', '#FFAA4C'),
             "HGAD": ("% of hydrologic group AD soil", 'Soil/Geology', '#B08457'),
             'prcp (mm/day)': ("Precipitation (mm/day)", 'Meteorological', '#FFAA4C'),
             'STRM_DENS': ("Stream density", "Hydrological", '#6B93D6'),
             "WDANN": ("Average days/year with precipitation", 'Meteorological', '#FFAA4C'),
             'q': ("Discharge (cfs)", "Hydrological", '#6B93D6'),
             "NLCD01_81": ("% Pasture/hay", 'Land Use / Land Cover', '#77B28C'),
             "ELEV_MIN": ('Minimum catchment elevation (m)', "Physical Attributes", '#7D7D7D'),
             'HGC': ('% of hydrologic group C soil', 'Soil/Geology', '#B08457'),
             'CONTACT': ("Subsurface flow contact time (d)", 'Soil/Geology', '#B08457'),
             'BEDPERM_6': ("% Bedrock: unconsolidated sand and gravel", 'Soil/Geology', '#B08457'),
             'RH': ('% Relative Humidity', 'Meteorological', '#FFAA4C'),
             'swe (kg/m2)': ("Snow-water equivalent (kg/m2)", 'Meteorological', '#FFAA4C'),
             'NLCD01_95': ("% Emergent wetlands", 'Land Use / Land Cover', '#77B28C'),
             'tmin (degrees C)': ("Minimum daily temperature (C)", 'Meteorological', '#FFAA4C'),
             'BFI': ("Baseflow index", "Hydrological", '#6B93D6'),
             'srad (W/m2)': ("Solar radiation (W/m2)", 'Meteorological', '#FFAA4C'),
             'SALINAVE': ('Mean salinity (mohms/cm)', "Hydrological", '#6B93D6'),
             'ARTIFICIAL': ('% Artificial flowpaths', "Hydrological", '#6B93D6'),
             'SANDAVE': ('Mean % sand', 'Soil/Geology', '#B08457'),
             "sinuosity": ("Mean stream sinuosity", "Hydrological", '#6B93D6'),
             'temp': ('Water temperature (C)', "Hydrological", '#6B93D6'),
             "vp (Pa)": ("Vapor pressure (Pa)", 'Meteorological', '#FFAA4C'),
             'tmax (degrees C)': ("Maximum daily temperature (C)", 'Meteorological', '#FFAA4C'),
             "SILTAVE": ("Mean % silt", 'Soil/Geology', '#B08457'),
             "lon": ('Longitude', "Physical Attributes", '#7D7D7D')}

#%% Random Forest for Feature selection
rf = RandomForestRegressor(random_state=0)
rf.fit(master_train, master_y_train)
feat_list = list(zip(master_train.columns, rf.feature_importances_))
feat_list.sort(key = lambda x: x[1])


#%% Plot random forest for floodplain
first_25 = feat_list[len(feat_list)-25:]

#%%
fig, ax = plt.subplots(figsize=(14, 10))
ax.barh([feat_dict[x[0]][0] for x in first_25], [x[1] for x in first_25 if x in feat],
          label = [feat_dict[x[0]][1] for x in first_25], 
          color = [feat_dict[x[0]][2] for x in first_25])
ax.set_title("Random Forest Floodplain Feature Importance for DO", fontsize=20)
ax.set_xlabel("Relative Feature Importance", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

# Custom legend
category_colors = {}
for key, value in feat_dict.items():
    category_colors[value[1]] = value[2]

handles = [Patch(color=color, label=category) for category, color in category_colors.items()]

# Add custom legend
ax.legend(handles=handles, loc='center right', title='Feature Category', title_fontsize=16, fontsize=14)

plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig(dir_path + "\\figures\\feature_selection\\FP_features_RF.png", dpi=300)
plt.show()
#%% RF on river data
rf_river = RandomForestRegressor(random_state=0, n_jobs=-1)
rf_river.fit(river_train, river_y_train)
feat_list_river = list(zip(river_train.columns, rf_river.feature_importances_))
feat_list_river.sort(key = lambda x: x[1])
#%% plot random forest for river
first25_river = feat_list_river[len(feat_list_river)-25:]
fig, ax = plt.subplots(figsize=(14, 10))
ax.barh([feat_dict[x[0]][0] for x in first25_river], [x[1] for x in first25_river],
         label = [feat_dict[x[0]][1] for x in first25_river], 
         color = [feat_dict[x[0]][2] for x in first25_river])
ax.set_title("Random Forest River Feature Importance for DO", fontsize=20)
ax.set_xlabel("Relative Feature Importance", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

# Custom legend
category_colors = {}
for key, value in feat_dict.items():
    category_colors[value[1]] = value[2]

handles = [Patch(color=color, label=category) for category, color in category_colors.items()]

# Add custom legend
ax.legend(handles=handles, loc='center right', title='Feature Category', title_fontsize=16, fontsize=14)

plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(dir_path + "\\figures\\feature_selection\\River_features_RF.png", dpi=300)
plt.show()
#%% Create a set of variables to keep based on Random Forest Feature selection
vars_to_keep = set()
for var in first25_river:
    vars_to_keep.add(var[0])
for var in first_25:
    vars_to_keep.add(var[0])

print(vars_to_keep)
#%% Save variables to keep to a text file
vars_to_keep.add('o')
with open(dir_path+"\\vars_to_keep_FS.txt", "w") as file:
    for item in vars_to_keep:
        file.write(item + "\n")