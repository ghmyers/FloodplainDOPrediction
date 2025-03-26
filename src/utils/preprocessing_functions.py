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
    data = data.replace(0, pd.NA)
    data -= epsilon
    df = pd.DataFrame(scaler.inverse_transform(data), columns=df_sc.columns, index=df_sc.index)

    return df


#%% Train/test split functions
def train_test_split_evenSites(df, split_pct):
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

def random_time_split(df, test_proportion, target_col):
    np.random.seed(random.randint(0, 1000))
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