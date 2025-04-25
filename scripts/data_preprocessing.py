"""
TITLE: Data Preprocessing
AUTHOR: Harrison Myers
DATE: 2025-04-01
DESCRIPTION: Takes river and floodplain data and preprocesses it. 
"""

#!/usr/bin/env python3
import os
import sys

# Set project directory dynamically
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)

# Ensure PROJECT_DIR is in sys.path for module imports
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Append src/ to Python's module search path
sys.path.append(os.path.join(PROJECT_DIR, "src"))

# Define subdirectories
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')

# Import logging
from src.utils.logging_setup import setup_logging
logger = setup_logging(log_dir="logs", log_filename="data_preprocessing.log")
logger.info(f"Script initialized: data_preprocessing.py")

# Import modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utils import preprocessing_functions as ppf
import numpy as np

# Read in floodplain data and drop any missing DO observations
OC2 = pd.read_csv(PROJECT_DIR + r"/data/OC2.csv", index_col=[0])
OC2.index = pd.to_datetime(OC2.index).date
LF1 = pd.read_csv(PROJECT_DIR + r"/data/LF1.csv", index_col=[0])
LF1.index = pd.to_datetime(LF1.index).date
LF2 = pd.read_csv(PROJECT_DIR + r"/data/LF2.csv", index_col=[0])
LF2.index = pd.to_datetime(LF2.index).date
OC4 = pd.read_csv(PROJECT_DIR + r"/data/OC4.csv", index_col=[0])
OC4.index = pd.to_datetime(OC4.index).date
LF3 = pd.read_csv(PROJECT_DIR + r"/data/LF3.csv", index_col=[0])
LF3.index = pd.to_datetime(LF3.index).date
OC1 = pd.read_csv(PROJECT_DIR + r"/data/OC1.csv", index_col=[0])
OC1.index = pd.to_datetime(OC1.index).date
OC3 = pd.read_csv(PROJECT_DIR + r"/data/OC3.csv", index_col=[0])
OC3.index = pd.to_datetime(OC3.index).date
site_dfs = [OC2, LF1, LF2, OC4, LF3, OC1, OC3]
OC2_noNA = OC2.dropna(subset=['o'])
LF1_noNA = LF1.dropna(subset=['o'])
LF2_noNA = LF2.dropna(subset=['o'])
OC4_noNA = OC4.dropna(subset=['o'])
LF3_noNA = LF3.dropna(subset=['o'])
OC1_noNA = OC1.dropna(subset=['o'])
OC3_noNA = OC3.dropna(subset=['o'])
site_dfs_noNA = [OC2_noNA, LF1_noNA, LF2_noNA, OC4_noNA, LF3_noNA, OC1_noNA, OC3_noNA]

# Read in river data and drop any missing DO observations
data = pd.read_csv(PROJECT_DIR + r"/data/river_training_data.csv", index_col=[0])
data.index = pd.to_datetime(data.index).date
data_noNA = data.dropna(subset=['o'])

# Apply random forest feature selection
with open(os.path.join(MODEL_DIR, 'old', 'vars_to_keep_FS.txt'), "r") as file:
    vars_to_keep = [line.strip() for line in file]
print(vars_to_keep)
for i, df in enumerate(site_dfs_noNA):
    df = df[vars_to_keep]
    df = df.sort_index(axis=1)
    site_dfs_noNA[i] = df
data_noNA = data_noNA[vars_to_keep]
master_fp = pd.concat(site_dfs_noNA)

# Only keep sites with more than a years worth of data
river_site_dfs = {}
data_noNA['site'] = data_noNA.groupby(['lat', 'lon']).ngroup()
sites = data_noNA['site'].unique()
for site in sites:
    temp_df = data_noNA[data_noNA['site'] == site]
    if len(temp_df) >= 365: # only keep sites with a year or more worth of data
        temp_df = temp_df.drop('site', axis=1)
        river_site_dfs[site] = temp_df
    else:
        data_noNA = data_noNA[data_noNA['site'] != site]
    river_site_dfs[site] = temp_df

# Drop site column
data_noNA.drop('site', axis=1, inplace=True)

# Sort features alphabetically
master_fp = master_fp.sort_index(axis=1)
data_noNA = data_noNA.sort_index(axis=1)

# Create scaler
# create a dataframe with USGS and site data for accurate scaling
all_data_df = pd.concat([master_fp, data_noNA])
all_data_df = all_data_df.sort_index(axis=1)

# First fit scaler just to y data for rescaling predictions
scaler_y = MinMaxScaler()
scaler_y.fit(np.array(all_data_df["o"]).reshape(-1, 1))

# Scaler for whole dataset
scaler = MinMaxScaler()
scaler.fit(all_data_df.values)

# Save scalers
import joblib
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_old.pkl'))
joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'y_scaler_old.pkl'))
    
    
# Define the river data pipeline
def river_pipeline(model_number = '1'):
    """
    Performs all necessary preprocessing steps to the river data for training and testing an LSTM.
    
    Args:
        model_number (str): String telling which config file to read hyperparameters from. Valid options are "1", "2", and "3".
    
    Returns:
        return_dict (dict): A dictionary containing the full scaled dataframe, the training and testing dataframes,
                            X_train, y_train, X_test, y_test, the training and testing generators, and the steps per epoch for training and testing. 
    """
    data = pd.read_csv(PROJECT_DIR + r"/data/river_training_data.csv", index_col=[0])
    data.index = pd.to_datetime(data.index).date
    
    # Drop any missing DO observations from the dataset
    data_noNA = data.dropna(subset=['o'])
    
    # Only keep features from random forest feature selection
    with open(os.path.join(MODEL_DIR, 'old', 'vars_to_keep_FS.txt'), "r") as file:
        vars_to_keep = [line.strip() for line in file]  
    print(vars_to_keep)
    data_noNA = data_noNA[vars_to_keep]
    
    # Drop sites with less than a years worth of data
    river_site_dfs = {}
    data_noNA['site'] = data_noNA.groupby(['lat', 'lon']).ngroup()
    sites = data_noNA['site'].unique()
    for site in sites:
        temp_df = data_noNA[data_noNA['site'] == site]
        if len(temp_df) >= 365: # only keep sites with a year or more worth of data
            temp_df = temp_df.drop('site', axis=1)
            river_site_dfs[site] = temp_df
        else:
            data_noNA = data_noNA[data_noNA['site'] != site]
        river_site_dfs[site] = temp_df
        
    data_noNA.drop('site', axis=1, inplace=True)
    
    assert len(river_site_dfs) == 480
    
    # Sort features alphabetically
    data_noNA = data_noNA.sort_index(axis=1)

    # Scale data from 0 to 1, add epsilon of 1e-7 to all values, then fill na's with 0
    data_sc_noNA = ppf.data_pipeline(data_noNA, 1e-7, scaler)
    
    # Read in necessary hyperparameters from config file
    import yaml
    with open(os.path.join(CONFIG_DIR, f'river_lstm{model_number}_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        
    window_length = config['window_length']
    batch_size = config['batch_size']
    split_pct = config['split_pct']
    seed = config['seed']
    
    # Split data into X_train, y_train, X_test, y_test
    data_sc_train, data_sc_test, riv_data_site_inds = ppf.train_test_split_evenSites(data_sc_noNA, split_pct, seed)
    X_train = data_sc_train.drop('o', axis=1)
    y_train = data_sc_train['o']
    X_test = data_sc_test.drop('o', axis=1)
    y_test = data_sc_test['o']
    
    # Get training and testing generators
    train_gen = ppf.data_generator(X_train, y_train, window_length, batch_size)
    test_gen = ppf.data_generator(X_test, y_test, window_length, batch_size)

    # Calculate steps per epoch
    steps_per_epoch = ppf.get_steps_per_epoch(X_train, window_length, batch_size)
    val_steps = ppf.get_steps_per_epoch(X_test, window_length, batch_size)
    
    return_dict = {
        "scaled_df": data_sc_noNA,
        "train_df": data_sc_train,
        "test_df": data_sc_test,
        "split_inds": riv_data_site_inds,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_gen": train_gen,
        "test_gen": test_gen,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps
    }
    return return_dict

def floodplain_pipeline(model_number="1"):
    """
    Performs all necessary preprocessing steps to the floodplain data for training and testing an LSTM.
    
    Args:
        model_number (str): The number of the model. Default is "1", other options are "2" and "3".
    
    Returns:
        return_dict (dict): A dictionary containing the full scaled dataframe, the training and testing dataframes,
                            X_train, y_train, X_test, y_test, the training and testing generators, and the steps per epoch for training and testing. 
    """
    # Read in floodplain data
    OC2 = pd.read_csv(PROJECT_DIR + r"/data/OC2.csv", index_col=[0])
    OC2.index = pd.to_datetime(OC2.index).date
    LF1 = pd.read_csv(PROJECT_DIR + r"/data/LF1.csv", index_col=[0])
    LF1.index = pd.to_datetime(LF1.index).date
    LF2 = pd.read_csv(PROJECT_DIR + r"/data/LF2.csv", index_col=[0])
    LF2.index = pd.to_datetime(LF2.index).date
    OC4 = pd.read_csv(PROJECT_DIR + r"/data/OC4.csv", index_col=[0])
    OC4.index = pd.to_datetime(OC4.index).date
    LF3 = pd.read_csv(PROJECT_DIR + r"/data/LF3.csv", index_col=[0])
    LF3.index = pd.to_datetime(LF3.index).date
    OC1 = pd.read_csv(PROJECT_DIR + r"/data/OC1.csv", index_col=[0])
    OC1.index = pd.to_datetime(OC1.index).date
    OC3 = pd.read_csv(PROJECT_DIR + r"/data/OC3.csv", index_col=[0])
    OC3.index = pd.to_datetime(OC3.index).date

    # Drop any missing DO observations
    OC2_noNA = OC2.dropna(subset=['o'])
    LF1_noNA = LF1.dropna(subset=['o'])
    LF2_noNA = LF2.dropna(subset=['o'])
    OC4_noNA = OC4.dropna(subset=['o'])
    LF3_noNA = LF3.dropna(subset=['o'])
    OC1_noNA = OC1.dropna(subset=['o'])
    OC3_noNA = OC3.dropna(subset=['o'])    
    
    # Concatenate into a list
    site_dfs_noNA = [OC2_noNA, LF1_noNA, LF2_noNA, OC4_noNA, LF3_noNA, OC1_noNA, OC3_noNA]
    
    # Drop columns that weren't selected by random forest feature selection
    # with open(os.path.join(OUTPUT_DIR, 'FS_11_vars_to_keep.txt'), "r") as file:
    #     vars_to_keep = [line.strip() for line in file]
    with open(os.path.join(MODEL_DIR, 'old', 'vars_to_keep_FS.txt'), "r") as file:
        vars_to_keep = [line.strip() for line in file]
        
    print(vars_to_keep)
    for i, df in enumerate(site_dfs_noNA):
        df = df[vars_to_keep]
        df = df.sort_index(axis=1)
        site_dfs_noNA[i] = df
    
    # Scale the site data 
    site_dfs_sc = []
    for df in site_dfs_noNA:
        df_sc = ppf.data_pipeline(df, 1e-7, scaler)
        site_dfs_sc.append(df_sc)
        
    # Get training and testing data
    # Split smoothed data into training and testing
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    # Read in necessary parameters from config file 
    import yaml
    with open(os.path.join(CONFIG_DIR, f'TL_lstm{model_number}_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        
    window_length = config['window_length']
    batch_size = config['batch_size']
    split_pct = config['split_pct']
    seed = config['seed']

    start_inds = []
    end_inds = []
    for df in site_dfs_sc:
        splits = ppf.random_time_split(df, split_pct, 'o', seed)
        X_train_list.append(splits[0])
        y_train_list.append(splits[1])
        X_test_list.append(splits[2])
        y_test_list.append(splits[3])
        start_inds.append(splits[4])
        end_inds.append(splits[5])
        
    # Save train/test split indices
    site_names = ['OC2', 'LF1', 'LF2', 'OC4', 'LF3', 'OC1', 'OC3']
    train_test_splits = list(zip(site_names, start_inds, end_inds))
    with open(PROJECT_DIR + f'/models/fp_train_test_split_indices_{model_number}.txt', 'w') as f:
        for site in train_test_splits:
            f.write(f"{site[0]}, {site[1]}, {site[2]}\n")

    X_train_master = pd.concat(X_train_list)
    y_train_master = pd.concat(y_train_list)
    X_test_master = pd.concat(X_test_list)
    y_test_master = pd.concat(y_test_list)

    # Generate training sequences
    train_gen_FP = ppf.data_generator(X_train_master, y_train_master, window_length, batch_size)
    test_gen_FP = ppf.data_generator(X_test_master, y_test_master, window_length, batch_size)
    
    # Get steps per epoch for training and testing datasets
    steps_per_epoch_FP = ppf.get_steps_per_epoch(X_train_master, window_length, batch_size)
    val_steps_FP = ppf.get_steps_per_epoch(X_test_master, window_length, batch_size)

    # Get X and y for each site for making predictions
    X_OC2, y_OC2 = ppf.reshape_data(site_dfs_sc[0].drop(['o'], axis=1), site_dfs_sc[0]['o'], window_length)
    X_LF1, y_LF1 = ppf.reshape_data(site_dfs_sc[1].drop(['o'], axis=1), site_dfs_sc[1]['o'], window_length)
    X_LF2, y_LF2 = ppf.reshape_data(site_dfs_sc[2].drop(['o'], axis=1), site_dfs_sc[2]['o'], window_length)
    X_OC4, y_OC4 = ppf.reshape_data(site_dfs_sc[3].drop(['o'], axis=1), site_dfs_sc[3]['o'], window_length)
    X_LF3, y_LF3 = ppf.reshape_data(site_dfs_sc[4].drop(['o'], axis=1), site_dfs_sc[4]['o'], window_length)
    X_OC1, y_OC1 = ppf.reshape_data(site_dfs_sc[5].drop(['o'], axis=1), site_dfs_sc[5]['o'], window_length)
    X_OC3, y_OC3 = ppf.reshape_data(site_dfs_sc[6].drop(['o'], axis=1), site_dfs_sc[6]['o'], window_length)
    X_master_rs = np.concatenate([X_OC2, X_LF1, X_LF2, X_OC4, X_LF3, X_OC1, X_OC3], axis=0)
    y_master_rs = np.concatenate([y_OC2, y_LF1, y_LF2, y_OC4, y_LF3, y_OC1, y_OC3])
    
    # Return start and end indices per site
    OC2_start = OC2_noNA.index[start_inds[0]]; OC2_end = OC2_noNA.index[end_inds[0]]
    LF1_start = LF1_noNA.index[start_inds[1]]; LF1_end = LF1_noNA.index[end_inds[1]]
    LF2_start = LF2_noNA.index[start_inds[2]]; LF2_end = LF2_noNA.index[end_inds[2]]
    OC4_start = OC4_noNA.index[start_inds[3]]; OC4_end = OC4_noNA.index[end_inds[3]]
    LF3_start = LF3_noNA.index[start_inds[4]]; LF3_end = LF3_noNA.index[end_inds[4]]
    OC1_start = OC1_noNA.index[start_inds[5]]; OC1_end = OC1_noNA.index[end_inds[5]]
    OC3_start = OC3_noNA.index[start_inds[6]]; OC3_end = OC3_noNA.index[end_inds[6]]

    return_dict = {
        'master': {
            "X_train_fp": X_train_master,
            "y_train_fp": y_train_master,
            "X_test_fp": X_test_master,
            "y_test_fp": y_test_master,
            "train_gen_fp": train_gen_FP,
            "test_gen_fp": test_gen_FP,
            "steps_per_epoch_fp": steps_per_epoch_FP,
            "val_steps_fp": val_steps_FP,
            "X_master_rs": X_master_rs,
            "y_master_rs": y_master_rs
        },
        "OC2": {
            "OC2_noNA": OC2_noNA,
            "X_OC2": X_OC2,
            "y_OC2": y_OC2,
            "OC2_start": OC2_start,
            "OC2_end": OC2_end
        },
        "LF1": {
            "LF1_noNA": LF1_noNA,
            "X_LF1": X_LF1,
            "y_LF1": y_LF1,
            "LF1_start": LF1_start,
            "LF1_end": LF1_end
        },
        "LF2": {
            "LF2_noNA": LF2_noNA,
            "X_LF2": X_LF2,
            "y_LF2": y_LF2,
            "LF2_start": LF2_start,
            "LF2_end": LF2_end
        },
        "OC4": {
            "OC4_noNA": OC4_noNA,
            "X_OC4": X_OC4,
            "y_OC4": y_OC4,
            "OC4_start": OC4_start,
            "OC4_end": OC4_end
        },
        "LF3": {
            "LF3_noNA": LF3_noNA,
            "X_LF3": X_LF3,
            "y_LF3": y_LF3,
            "LF3_start": LF3_start,
            "LF3_end": LF3_end
        },
        "OC1": {
            "OC1_noNA": OC1_noNA,
            "X_OC1": X_OC1,
            "y_OC1": y_OC1,
            "OC1_start": OC1_start,
            "OC1_end": OC1_end
        },
        "OC3": {
            "OC3_noNA": OC3_noNA,
            "X_OC3": X_OC3,
            "y_OC3": y_OC3,
            "OC3_start": OC3_start,
            "OC3_end": OC3_end
        }  
    }
    
    return return_dict

def ADDA_pipeline(model_number = '1'):
    """Performs all necessary preprocessing steps to the floodplain and rive data for training and testing an LSTM with Adversarial Discriminative Domain Adaptation.

    Args:
        model_number (str, optional): Defaults to '1'.

    Returns:
        return_dict (dict): A dictionary containing the full scaled dataframe, the training and testing dataframes,
                            X_train, y_train, X_test, y_test, the training and testing generators, and the steps per epoch for training and testing.
    """
    # Preprocess river data
    data = pd.read_csv(PROJECT_DIR + r"/data/river_training_data.csv", index_col=[0])
    data.index = pd.to_datetime(data.index).date
         
    # Drop any missing DO observations from the dataset
    data_noNA = data.dropna(subset=['o'])
    
    # Only keep features from random forest feature selection
    with open(os.path.join(MODEL_DIR, 'old', 'vars_to_keep_FS.txt'), "r") as file:
        vars_to_keep = [line.strip() for line in file]  
    print(vars_to_keep)
    # vars_to_keep.append('domain')
    data_noNA = data_noNA[vars_to_keep]
    
    # Drop sites with less than a years worth of data
    river_site_dfs = {}
    data_noNA['site'] = data_noNA.groupby(['lat', 'lon']).ngroup()
    sites = data_noNA['site'].unique()
    for site in sites:
        temp_df = data_noNA[data_noNA['site'] == site]
        if len(temp_df) >= 365: # only keep sites with a year or more worth of data
            temp_df = temp_df.drop('site', axis=1)
            river_site_dfs[site] = temp_df
        else:
            data_noNA = data_noNA[data_noNA['site'] != site]
        river_site_dfs[site] = temp_df
        
    data_noNA.drop('site', axis=1, inplace=True)
    
    assert len(river_site_dfs) == 480
    
    # Sort features alphabetically
    data_noNA = data_noNA.sort_index(axis=1)

    # Scale data from 0 to 1, add epsilon of 1e-7 to all values, then fill na's with 0
    data_sc_noNA = ppf.data_pipeline(data_noNA, 1e-7, scaler)
    
    # Read in necessary hyperparameters from config file
    import yaml
    with open(os.path.join(CONFIG_DIR, f'adda_lstm{model_number}_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        
    window_length = config['window_length']
    batch_size = config['batch_size']
    split_pct = config['split_pct']
    seed = config['seed']
    
    # Split data into X_train, y_train, X_test, y_test
    data_sc_train, data_sc_test, riv_data_site_inds = ppf.train_test_split_evenSites(data_sc_noNA, split_pct, seed)
    X_train = data_sc_train.drop(columns=['o',])
    y_train = data_sc_train['o']
    # domain_y_train_series = pd.Series(np.full(len(y_train), 0), name='domain')
    # y_train = pd.concat([y_train, domain_y_train_series], axis=1)
    y_train = pd.DataFrame({'o': y_train, 'domain': pd.Series(0, index=y_train.index)})
    # y_train['domain'] = np.full(len(y_train), 0)
    X_test = data_sc_test.drop(columns=['o'])
    y_test = data_sc_test['o']
    # domain_y_test_series = pd.Series(np.full(len(y_test), 0), name='domain')
    # y_test = pd.concat([y_test, domain_y_test_series], axis=1)
    y_test = pd.DataFrame({'o': y_test, 'domain': pd.Series(0, index=y_test.index)})
    # y_test['domain'] = np.full(len(y_test), 0)
    
    ### Preprocess floodplain data
    # Read in floodplain data
    OC2 = pd.read_csv(PROJECT_DIR + r"/data/OC2.csv", index_col=[0])
    OC2.index = pd.to_datetime(OC2.index).date
    LF1 = pd.read_csv(PROJECT_DIR + r"/data/LF1.csv", index_col=[0])
    LF1.index = pd.to_datetime(LF1.index).date
    LF2 = pd.read_csv(PROJECT_DIR + r"/data/LF2.csv", index_col=[0])
    LF2.index = pd.to_datetime(LF2.index).date
    OC4 = pd.read_csv(PROJECT_DIR + r"/data/OC4.csv", index_col=[0])
    OC4.index = pd.to_datetime(OC4.index).date
    LF3 = pd.read_csv(PROJECT_DIR + r"/data/LF3.csv", index_col=[0])
    LF3.index = pd.to_datetime(LF3.index).date
    OC1 = pd.read_csv(PROJECT_DIR + r"/data/OC1.csv", index_col=[0])
    OC1.index = pd.to_datetime(OC1.index).date
    OC3 = pd.read_csv(PROJECT_DIR + r"/data/OC3.csv", index_col=[0])
    OC3.index = pd.to_datetime(OC3.index).date

    # Drop any missing DO observations
    OC2_noNA = OC2.dropna(subset=['o'])
    LF1_noNA = LF1.dropna(subset=['o'])
    LF2_noNA = LF2.dropna(subset=['o'])
    OC4_noNA = OC4.dropna(subset=['o'])
    LF3_noNA = LF3.dropna(subset=['o'])
    OC1_noNA = OC1.dropna(subset=['o'])
    OC3_noNA = OC3.dropna(subset=['o'])    
    
    # Concatenate into a list
    site_dfs_noNA = [OC2_noNA, LF1_noNA, LF2_noNA, OC4_noNA, LF3_noNA, OC1_noNA, OC3_noNA]
    
    for i, df in enumerate(site_dfs_noNA):
        df = df[vars_to_keep]
        df = df.sort_index(axis=1)
        site_dfs_noNA[i] = df
    
    # Scale the site data 
    site_dfs_sc = []
    for df in site_dfs_noNA:
        df_sc = ppf.data_pipeline(df, 1e-7, scaler)
        site_dfs_sc.append(df_sc)
        
    # Get training and testing data
    # Split smoothed data into training and testing
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    start_inds = []
    end_inds = []
    for df in site_dfs_sc:
        splits = ppf.random_time_split(df, split_pct, 'o', seed)
        X_train_list.append(splits[0])
        y_train_list.append(splits[1])
        X_test_list.append(splits[2])
        y_test_list.append(splits[3])
        start_inds.append(splits[4])
        end_inds.append(splits[5])
        
    # Save train/test split indices
    site_names = ['OC2', 'LF1', 'LF2', 'OC4', 'LF3', 'OC1', 'OC3']
    train_test_splits = list(zip(site_names, start_inds, end_inds))
    with open(PROJECT_DIR + f'/models/fp_train_test_split_indices_{model_number}.txt', 'w') as f:
        for site in train_test_splits:
            f.write(f"{site[0]}, {site[1]}, {site[2]}\n")

    X_train_master = pd.concat(X_train_list)
    y_train_master = pd.concat(y_train_list)
    # Add domain label to y_train
    # y_train_master = pd.concat([y_train_master, domain_train_series], axis=1)
    y_train_master = pd.DataFrame({'o': y_train_master, 'domain': pd.Series(1, index=y_train_master.index)})
    X_test_master = pd.concat(X_test_list)
    y_test_master = pd.concat(y_test_list)
    # Add domain label to y_test
    # y_test_master = pd.concat([y_test_master, domain_test_series], axis=1)
    y_test_master = pd.DataFrame({'o': y_test_master, 'domain': pd.Series(1, index=y_test_master.index)})

    # Combine river and floodplain training and testing dataframes into one
    X_train_master = pd.concat([X_train, X_train_master])
    X_test_master = pd.concat([X_test, X_test_master])
    y_train_master = pd.concat([y_train, y_train_master])
    y_test_master = pd.concat([y_test, y_test_master])
    

    # Get training and testing generators
    train_gen = ppf.data_generator_adda_balanced(X_train_master, y_train_master, window_length, batch_size)
    test_gen = ppf.data_generator_adda_balanced(X_test_master, y_test_master, window_length, batch_size)

    # Calculate steps per epoch
    steps_per_epoch = ppf.get_steps_per_epoch(X_train_master, window_length, batch_size)
    val_steps = ppf.get_steps_per_epoch(X_test_master, window_length, batch_size)
    
    return_dict = {
        "scaled_df": data_sc_noNA,
        "train_df": data_sc_train,
        "test_df": data_sc_test,
        "split_inds": riv_data_site_inds,
        "X_train": X_train_master,
        "y_train": y_train_master,
        "X_test": X_test_master,
        "y_test": y_test_master,
        "train_gen": train_gen,
        "test_gen": test_gen,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps
    }
    return return_dict
