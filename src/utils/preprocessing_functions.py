from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import pandas as pd

#%% Data pipelines for scaling data, adding an epsilon, and filling NAs with 0s
def data_pipeline(df, epsilon, scaler):
    """
    Scales data using a pre-fit scaler, adds epsilon to the whole dataframe (to avoid taking the log of 0), then fills na's with 0.
    """
    df_sc = pd.DataFrame(scaler.transform(df.values), columns=df.columns, index=df.index)
    df_sc += epsilon
    df_sc = df_sc.fillna(0)

    return df_sc

def reverse_pipeline(df_sc, epsilon, scaler):
    """
    Applies inverse transformations to reverse scaled data back to its original state
    """
    data -= epsilon
    df = pd.DataFrame(scaler.inverse_transform(data), columns=df_sc.columns, index=df_sc.index)

    return df


#%% Train/test split functions
def train_test_split_evenSites(df, split_pct, seed):
    np.random.seed(seed)
    # Create site column based on unique latitude and longitude tuples
    df['site'] = df.groupby(['lat', 'lon']).ngroup()
    sites = df['site'].unique()
    train_splits = []
    test_splits  = []
    split_inds = []
    for site in sites:
        temp_df = df[df['site'] == site]
        split_ind = int(np.floor((1-split_pct)*int(len(temp_df))))
        split_inds.append(split_ind)
        start_ind = np.random.randint(0, len(temp_df) - split_ind)
        end_ind = start_ind + split_ind
        test_df = temp_df.iloc[start_ind:end_ind, :]
        train_df = pd.concat([temp_df.iloc[:start_ind, :], temp_df.iloc[end_ind:, :]])
        train_splits.append(train_df)
        test_splits.append(test_df)

    # Zip lists together, shuffle them, then unzip them
    zipped_list = list(zip(train_splits, test_splits))
    random.shuffle(zipped_list)
    train_splits, test_splits = zip(*zipped_list)
    
    Train = pd.concat(train_splits)
    Test = pd.concat(test_splits)
    
    Train.drop("site", axis=1, inplace=True)
    Test.drop('site', axis=1, inplace=True)
    
    return Train, Test, split_inds

def random_time_split(df, test_proportion, target_col, seed):
    np.random.seed(seed)
    # Calculate the number of test samples
    test_size = int(np.floor(len(df) * (1- test_proportion)))
    
    # Randomly select a starting index for the test set
    start_idx = np.random.randint(0, len(df) - test_size)
    end_idx = start_idx + test_size
    
    # Define the train and test sets
    test_df = df[start_idx:start_idx + test_size]
    train_df = pd.concat([df[:start_idx], df[start_idx + test_size:]])

    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test, start_idx, end_idx

def train_test_split(df, split_pct, target_col):
    split_ind = int(np.floor(len(df)* split_pct))
    end_idx = len(df) - 1
    train_df = df[:split_ind]
    test_df = df[split_ind:]
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    return X_train, y_train, X_test, y_test, split_ind, end_idx

# Randomly sample x% of the data
def rand_sample(df, sample_pct):
    df['site'] = df.groupby(['lat', 'lon']).ngroup()
    sites = df['site'].unique()
    df_samples = []
    for site in sites:
        temp_df = df[df['site'] == site]
        split_len = round(len(temp_df)*sample_pct)
        start_ind = random.randint(0, len(temp_df) - split_len)
        slice = temp_df.iloc[start_ind:start_ind+split_len]
        slice = slice.drop('site', axis=1)
        df_samples.append(slice)
    random.shuffle(df_samples)
    rand_sampled_df = pd.concat(df_samples)

    return rand_sampled_df

#%% Functions to reshape data and store locally or use a generator
# Stores sequences locally
def reshape_data(X, y, window_length):
    # Convert X and y to numpy arrays
    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)
    
    n_samples = len(X)
    n_sequences = n_samples - window_length + 1
    
    # Lists to hold the valid sequences
    valid_sequences = []
    valid_targets = []
    target_indices = []
    
    indices = np.arange(n_sequences)
    
    for i in indices:
        # Get start and end dates of sequence
        sequence_start = X.index[i]
        sequence_end = X.index[i + window_length - 1]
        
        # Check if the sequence is valid (comes from the same site)
        if (sequence_end - sequence_start).days == window_length - 1:
            X_seq = X_array[i:i + window_length - 1]
            y_seq = y_array[i + window_length - 1]
            
            valid_sequences.append(X_seq)
            valid_targets.append(y_seq)
            
            # Store the corresponding indices for the sequence and the target
            target_indices.append(X.index[i + window_length - 1])
    
    # Convert lists to numpy arrays
    X = np.array(valid_sequences)
    y = y[target_indices]
    
    return X, y

# Define the generator function for generating sequences to feed into LSTM
def data_generator(X, y, window_length, batch_size, dropNA_batches=False, na_pct=0.75):
    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)
    n_samples = len(X)

    while True:
        indices = np.arange(n_samples - window_length + 1)
        X_batch = []
        y_batch = []
        
        # for site, indices in valid_sequences.items():
        for i in indices:
            # Get start and end dates of sequence
            sequence_start = X.index[i]
            sequence_end = X.index[i + window_length - 1]
            if (sequence_end - sequence_start).days == window_length - 1:
                X_seq = X_array[i:i + window_length]
                y_seq = y_array[i + window_length - 1]
                
                X_batch.append(X_seq)
                y_batch.append(y_seq)
                
                if len(X_batch) == batch_size:
                    y_batch_array = np.array(y_batch)
                    if dropNA_batches:
                        # Check if na_pct or more of the y_batch is missing
                        na_count = np.count_nonzero(np.isnan(y_batch_array))
                        if na_count / batch_size >= na_pct:
                            # Reset batches and continue to the next iteration
                            X_batch, y_batch = [], []
                            continue
                    yield np.array(X_batch), np.array(y_batch)
                    X_batch, y_batch = [], []
            
        if X_batch and y_batch:  # If there are any remaining sequences not yielded yet
            yield np.array(X_batch), np.array(y_batch)
            
def data_generator_adda(X, y, window_length, batch_size, dropNA_batches=False, na_pct=0.75):
    X_array = X.to_numpy(dtype=np.float32)
    y_do = y['o'].values.astype(np.float32)
    y_domain = y['domain'].values.astype(np.float32)
    n_samples = len(X)

    while True:
        indices = np.arange(n_samples - window_length + 1)
        X_batch = []
        y_do_batch = []
        y_domain_batch = []

        for i in indices:
            sequence_start = X.index[i]
            sequence_end = X.index[i + window_length - 1]

            if (sequence_end - sequence_start).days == window_length - 1:
                X_seq = X_array[i:i + window_length]
                y_do_seq = y_do[i + window_length - 1]
                y_domain_seq = y_domain[i + window_length - 1]

                X_batch.append(X_seq)
                y_do_batch.append(y_do_seq)
                y_domain_batch.append(y_domain_seq)

                if len(X_batch) == batch_size:
                    y_batch_array_do = np.array(y_do_batch)
                    if dropNA_batches:
                        na_count = np.count_nonzero(np.isnan(y_batch_array_do))
                        if na_count / batch_size >= na_pct:
                            X_batch, y_do_batch, y_domain_batch = [], [], []
                            continue
                    yield (
                        np.array(X_batch),
                        {
                            "do_output": y_batch_array_do.reshape(-1, 1),
                            "domain_output": np.array(y_domain_batch).reshape(-1, 1)
                        }
                    )
                    X_batch, y_do_batch, y_domain_batch = [], [], []

        # Yield final partial batch (optional)
        if X_batch:
            yield (
                np.array(X_batch),
                {
                    "do_output": np.array(y_do_batch).reshape(-1, 1),
                    "domain_output": np.array(y_domain_batch).reshape(-1, 1)
                }
            )

def data_generator_adda_balanced(X, y, window_length, batch_size,
                        source_fraction=0.8, dropNA_batches=False, na_pct=0.75):
    # Convert to numpy
    X_array = X.to_numpy(dtype=np.float32)
    y_do = y['o'].values.astype(np.float32)
    y_domain = y['domain'].values.astype(np.float32)
    indices = np.arange(len(X) - window_length + 1)

    # Separate valid sequence start indices by domain
    valid_src_idx = []
    valid_tar_idx = []

    for i in indices:
        sequence_start = X.index[i]
        sequence_end = X.index[i + window_length - 1]

        if (sequence_end - sequence_start).days == window_length - 1:
            domain_val = y_domain[i + window_length - 1]
            if domain_val == 0:
                valid_src_idx.append(i)
            elif domain_val == 1:
                valid_tar_idx.append(i)

    source_batch_size = int(batch_size * source_fraction)
    target_batch_size = batch_size - source_batch_size

    while True:
        X_batch, y_do_batch, y_domain_batch = [], [], []

        if len(valid_tar_idx) == 0:
            raise ValueError("No target (domain=1) sequences found!")

        # Random sampling (no replacement for source, with replacement for target)
        src_sampled = np.random.choice(valid_src_idx, size=source_batch_size, replace=False)
        tar_sampled = np.random.choice(valid_tar_idx, size=target_batch_size, replace=True)

        all_sampled = np.concatenate([src_sampled, tar_sampled])
        np.random.shuffle(all_sampled)

        for i in all_sampled:
            X_seq = X_array[i:i + window_length]
            y_do_seq = y_do[i + window_length - 1]
            y_domain_seq = y_domain[i + window_length - 1]

            X_batch.append(X_seq)
            y_do_batch.append(y_do_seq)
            y_domain_batch.append(y_domain_seq)

        y_batch_array_do = np.array(y_do_batch)
        if dropNA_batches:
            na_count = np.count_nonzero(np.isnan(y_batch_array_do))
            if na_count / batch_size >= na_pct:
                continue

        yield (
            np.array(X_batch),
            {
                "do_output": y_batch_array_do.reshape(-1, 1),
                "domain_output": np.array(y_domain_batch).reshape(-1, 1)
            }
        )




def get_steps_per_epoch(X, window_length, batch_size):
    """A function to count the number of valid sequences in the training data, X

    Args:
        X (pd.DataFrame or np.array): The training data
        window_length (int): The window length for input to the LSTM
        batch_size (int): The batch size for the LSTM

    Returns:
        int: the number of steps per epoch
    """
    n_samples = len(X)
    indices = np.arange(n_samples - window_length + 1)
    valid_sequence_count = 0

    for i in indices:
        sequence_start = X.index[i]
        sequence_end = X.index[i + window_length - 1]
        if (sequence_end - sequence_start).days == window_length - 1:
            valid_sequence_count += 1
    print(valid_sequence_count)
    steps_per_epoch = valid_sequence_count // batch_size
    
    return steps_per_epoch

### RANDOM FOREST FEATURE SELECTION ###
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib
import os

def rf_feature_selection(fp_data, river_data, target, seed, feat_dict, MODEL_DIR=None, n_features=25,
                        savefig=False, fig_fpath=None, feature_fpath=None, performance_fpath=None):
    """
    Function to perform random forest feature selection, plot the results, and save the 
    selected features to a text file. Tests the random forest predictor and stores results as a .txt file.
    
    Args:
        fp_data (pd.DataFrame): The input data from the floodplain domain.
        river_data (pd.DataFrame): The input data from the river domain.
        target (pd.Series): The target variable.
        seed (int): The random seed for reproducibility.
        feat_dict (dict): A dictionary mapping feature names to their importance scores.
        savefig (bool): Whether to save the feature importance plot. Default is False.
        MODEL_DIR (str): Path to directory for saving trained models.
        fig_fpath (str): The file path to save the figure to. Default is None.
        feature_fpath (str): The file path to save the selected features to. Default is None.
        performance_fpath (str): The file path to save the random forest performance to. Default is None.
        
    Returns:
        None
    """
    # First scale the floodplaindata from 0 to 1
    fp_scaler = MinMaxScaler()
    fp_scaler.fit(fp_data)
    fp_data_sc = pd.DataFrame(fp_scaler.transform(fp_data), columns=fp_data.columns, index=fp_data.index)
    
    # Fit target scaler for model evaluation
    fp_y_scaler = MinMaxScaler()
    fp_y_scaler.fit(np.array(fp_data[target]).reshape(-1, 1))
        
    # River data
    riv_scaler = MinMaxScaler()
    riv_scaler.fit(river_data)
    river_data_sc = pd.DataFrame(riv_scaler.transform(river_data), columns=river_data.columns, index=river_data.index)
    
    # Fit target scaler for model evaluation
    riv_y_scaler = MinMaxScaler()
    riv_y_scaler.fit(np.array(river_data[target]).reshape(-1, 1))  
    
    # Get site column using latitude and longitude
    fp_data_sc['site'] = fp_data_sc.groupby(['lat', 'lon']).ngroup()
    river_data_sc['site'] = river_data_sc.groupby(['lat', 'lon']).ngroup()
    
    # Drop NAs from dataframe
    fp_data_sc.dropna(inplace=True)
    river_data_sc.dropna(inplace=True)
    
    # Index dataframe by site, and perform a custom train-test split that preserves equal proportions of training/testing data across sites
    # def train_test_split_ES(data, split_pct):
    #     sites = data['site'].unique()
    #     train_splits = []
    #     test_splits  = []
    #     split_inds = []
    #     for site in sites:
    #         temp_df = data[data['site'] == site]
    #         split_ind = int(np.floor(split_pct*int(len(temp_df))))
    #         split_inds.append(split_ind)
    #         train_df = temp_df.iloc[:split_ind, :]
    #         test_df = temp_df.iloc[split_ind:, :]
    #         train_splits.append(train_df)
    #         test_splits.append(test_df)
    #     train = pd.concat(train_splits)
    #     test = pd.concat(test_splits)
        
    #     return train, test
    
    # Split data into training and testing and drop site column
    # Floodplain
    fp_train, fp_test = train_test_split_evenSites(fp_data_sc, 0.8, seed)
    fp_train.drop("site", axis=1, inplace=True)
    fp_test.drop("site", axis=1, inplace=True)
    
    # River
    river_train, river_test = train_test_split_evenSites(river_data_sc, 0.8, seed)
    river_train.drop("site", axis=1, inplace=True)
    river_test.drop("site", axis=1, inplace=True)
    
    # Get X, y
    # Floodplain
    X_train_fp = fp_train.drop(target, axis=1)
    y_train_fp = fp_train[target]
    X_test_fp = fp_test.drop(target, axis=1)
    y_test_fp = fp_test[target]
    
    # River
    X_train_riv = river_train.drop(target, axis=1)
    y_train_riv = river_train[target]
    X_test_riv = river_test.drop(target, axis=1)
    y_test_riv = river_test[target]
    
    # Perform random forest feature selection for the floodplain data
    # Floodplain
    try:
        rf_fp = joblib.load(MODEL_DIR + "/RF/rf_floodplain_model.pkl")
        rf_riv = joblib.load(MODEL_DIR + "/RF/rf_river_model.pkl")
        
    except Exception as e:
        print(e)
        # Floodplain
        rf_fp = RandomForestRegressor(random_state=seed)
        rf_fp.fit(X_train_fp, y_train_fp)
        
        # River
        rf_riv = RandomForestRegressor(random_state=seed)
        rf_riv.fit(X_train_riv, y_train_riv)
        
        # Save trained random forest models to directory
        # Save the floodplain RF model
        joblib.dump(rf_fp, os.path.join(MODEL_DIR, "rf_floodplain_model.pkl"))

        # Save the river RF model
        joblib.dump(rf_riv, os.path.join(MODEL_DIR, "rf_river_model.pkl")) 
    
        print(f"Random Forest models saved in {MODEL_DIR}")

    # Get feature importances
    feat_list = list(zip(X_train_fp.columns, rf_fp.feature_importances_))
    feat_list.sort(key = lambda x: x[1])
    features_fp = feat_list[len(feat_list)-n_features:]
    
    
    feat_list = list(zip(X_train_riv.columns, rf_riv.feature_importances_))
    feat_list.sort(key = lambda x: x[1])
    features_riv = feat_list[len(feat_list)-n_features:]
    
    # Plot important river and floodplain features side by side
    def plot_side_by_side_feature_importance(features_fp, features_riv, feat_dict, 
                                         savefig=False, fig_fpath=None):
        """
        Plots side-by-side horizontal bar charts of feature importances 
        from floodplain and river Random Forests.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        # Prepare data for each subplot
        for ax, feat_list, title in zip(axes, 
                                        [features_fp, features_riv], 
                                        [f"Floodplain Feature Importance (Top {n_features})", f"River Feature Importance (Top {n_features})"]):
            feature_names = [feat_dict[x[0]][0] for x in feat_list]
            importances = [x[1] for x in feat_list]
            colors = [feat_dict[x[0]][2] for x in feat_list]

            ax.barh(feature_names, importances, color=colors)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Relative Importance", fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            # ax.invert_yaxis()  # highest importance at top
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Build legend
        category_colors = {v[1]: v[2] for v in feat_dict.values()}
        handles = [Patch(color=color, label=cat) for cat, color in category_colors.items()]
        axes[1].legend(handles=handles, loc='lower right', title="Feature Category", fontsize=12, title_fontsize=13)

        plt.tight_layout()
        
        if savefig:
            plt.savefig(fig_fpath, dpi=300)
        
        plt.show()
        
    # Save a set of the top 20 most important variables
    vars_to_keep = set()
    for var in features_riv:
        vars_to_keep.add(var[0])
    for var in features_fp:
        vars_to_keep.add(var[0])
    print(f"Total number of features to keep: {len(vars_to_keep)}")
    print(vars_to_keep)
    
    vars_to_keep.add('o')
    with open(feature_fpath, "w") as file:
        for item in vars_to_keep:
            file.write(item + "\n")
    
    # Plot feature importance side by side
    plot_side_by_side_feature_importance(features_fp, features_riv, feat_dict, savefig=True, fig_fpath=fig_fpath)
    
    # Evaluate Random Forest model performance 
    import hydroeval as he
    from sklearn.metrics import mean_squared_error, r2_score
    import math
    
    # Floodplain
    fp_preds = rf_fp.predict(X_test_fp)
    fp_preds = fp_y_scaler.inverse_transform(np.array(fp_preds).flatten().reshape(-1, 1))
    y_test_fp = fp_y_scaler.inverse_transform(np.array(y_test_fp).flatten().reshape(-1, 1))
    rmse_fp = math.sqrt(mean_squared_error(y_test_fp, fp_preds, squared=False))
    r2_fp = r2_score(y_test_fp, fp_preds)
    kge_fp = he.evaluator(he.kge, y_test_fp, fp_preds)
    bias_fp = he.evaluator(he.pbias, y_test_fp, fp_preds)
    
    # River
    riv_preds = rf_riv.predict(X_test_riv)
    riv_preds = riv_y_scaler.inverse_transform(np.array(riv_preds).flatten().reshape(-1, 1))
    y_test_riv = riv_y_scaler.inverse_transform(np.array(y_test_riv).flatten().reshape(-1, 1))
    rmse_riv = math.sqrt(mean_squared_error(y_test_riv, riv_preds, squared=False))
    r2_riv = r2_score(y_test_riv, riv_preds)
    kge_riv = he.evaluator(he.kge, y_test_riv, riv_preds)
    bias_riv = he.evaluator(he.pbias, y_test_riv, riv_preds)
    
    performance_metrics = {"Floodplain Random Forest RMSE:": round(rmse_fp, 2),
                           'Floodplain Random Forest KGE:': round(kge_fp[0][0], 2),
                           f"Floodplain Random Forest R{chr(0x00B2)}:": round(r2_fp, 2),
                           'Floodplain Random Forest Bias:': round(bias_fp[0], 2),
                           "River Random Forest RMSE:": round(rmse_riv, 2),
                           'River Random Forest KGE:': round(kge_riv[0][0], 2),
                           f"River Random Forest R{chr(0x00B2)}:": round(r2_riv, 2),
                           'River Random Forest Bias:': round(bias_riv[0], 2)}

    # Save performance metrics
    with open(performance_fpath, "w") as f:
        for k, v in performance_metrics.items():
            f.write(f"{k} {v}\n")


    
### RANDOM FOREST PREDICTIONS TO FILL TARGET VARIABLE #####
def fill_missing_w_rf_predictions(data, target, model_path, flag_col="rf_pred"):
    """Loads a trained RF model and fills missing values in the target column using predictions.
 
    Args:
        data (pd.DataFrame): DataFrame with missing values.
        target (str): Column name of the target variable.
        model_path (str): Path to trained RF model.
        flag_col (str): Name of the flag column for predicted values.
    
    Returns:
        pd.DataFrame: DataFrame with missing target values filled and predictions flagged.
    """
    # Drop rows where target is NaN for scaling
    data_no_nan = data.dropna(subset=[target])
    
    # Fit scaler only on valid data (as during training)
    scaler = MinMaxScaler()
    scaler.fit(data_no_nan.drop(columns=[flag_col], errors='ignore'))  # Drop flag_col if exists
    
    # Scale full data (fillna won't interfere with transform)
    data_sc = pd.DataFrame(scaler.transform(data.drop(columns=[flag_col], errors='ignore')),
                           columns=[col for col in data.columns if col != flag_col],
                           index=data.index)

    # Target scaler (if you need it later for inverse transform of just y)
    y_scaler = MinMaxScaler()
    y_scaler.fit(np.array(data_no_nan[target]).reshape(-1, 1))
    
    # Load trained model
    model = joblib.load(model_path)
    trained_features = model.feature_names_in_.tolist()  # Sklearn 1.0+

    # Prepare working DataFrame
    df_out = data_sc.copy()
    df_out[flag_col] = 0
    
    # Find missing target rows
    missing_mask = data[target].isna()

    if missing_mask.sum() == 0:
        print("No missing values to fill.")
        data[flag_col] = 0
        return data

    # Get predictors for missing values (same features, same order as model)
    X_missing = df_out.loc[missing_mask, trained_features]

    # Predict
    preds = model.predict(X_missing)

    # Fill and flag
    df_out.loc[missing_mask, target] = preds
    df_out.loc[missing_mask, flag_col] = 1

    # Inverse scale everything except the flag column
    flag_series = df_out[flag_col]
    df_out_noflag = df_out.drop(columns=[flag_col])
    df_out_inv = pd.DataFrame(scaler.inverse_transform(df_out_noflag),
                              columns=df_out_noflag.columns,
                              index=df_out_noflag.index)
    df_out_inv[flag_col] = flag_series

    return df_out_inv

    
    
