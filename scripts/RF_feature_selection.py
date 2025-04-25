#%%
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

PROJ_DIR = os.path.dirname(os.getcwd())
os.chdir(PROJ_DIR)
DATA_DIR = PROJ_DIR + "/data"
OUTPUT_DIR = os.path.join(PROJ_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(PROJ_DIR, "models")
#%%
river_data = pd.read_csv(DATA_DIR + "/river_training_data.csv", index_col=[0])
river_data.index = pd.to_datetime(river_data.index).date

site_data = []
for f in os.listdir(DATA_DIR):
    if f.startswith('river'):
        pass
    elif f.endswith('.csv'):
        df = pd.read_csv(DATA_DIR + "/" + f, index_col=[0])
        df.index = pd.to_datetime(df.index).date
        site_data.append(df)
    else:
        pass
fp_data = pd.concat(site_data)
fp_data.sort_index(axis=1, inplace=True)
river_data.sort_index(axis=1, inplace=True)

# Create dictionary of variable names with descriptions, labels, and plotting colors
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
             "lon": ('Longitude', "Physical Attributes", '#7D7D7D'),
             "HGA": ('% of hydrologic group A soil', 'Soil/Geology', '#B08457'),
             "MAXP6190": ("Watershed max annual precipitation (mm)", "Meteorological", "#FFAA4C"),
             "LSTFZ6190": ("Average mean day of the last freeze", 'Meteorological', '#FFAA4C'),
             "BASIN_SLOPE": ("Catchment average slope", "Physical Attributes", '#7D7D7D'),
             "EWT": ("Average depth to water table", "Soil/Geology", '#B08457'),
             "HGB": ('% of hydrologic group B soil', 'Soil/Geology', '#B08457'),
             "BEDPERM_7": ("% of bedrock classified as Carbonate rock", "Soil/Geology", '#B08457'),
             "P97": ("Phosphorus from fertilizer/manure (1997)", 'Land Use / Land Cover', '#77B28C'),
             "PRSNOW": ("% of precipitation as snow", 'Meteorological', '#FFAA4C'),
             "PEST219": ("Estimate of agricultural pesticide use (kg/sq. km)", 'Land Use / Land Cover', '#77B28C'),
             "PET": ("Mean annual potential evapotranspiration (mm)", 'Meteorological', '#FFAA4C'),
             "NLCD01_11": ("% Emergent herbaceous wetlands", 'Land Use / Land Cover', '#77B28C')}

import sys
sys.path.append(os.path.join(PROJ_DIR, "src"))
from utils.preprocessing_functions import rf_feature_selection

rf_feature_selection(fp_data, river_data, 'o', 0, feat_dict, MODEL_DIR, savefig=True, n_features=25,
                     fig_fpath=os.path.join(FIGURE_DIR, "RF_25_features_barplot.png"),
                     feature_fpath=os.path.join(OUTPUT_DIR, "FS_25_vars_to_keep.txt"),
                     performance_fpath=os.path.join(OUTPUT_DIR, "RF_model_performance.txt"))

# %%
