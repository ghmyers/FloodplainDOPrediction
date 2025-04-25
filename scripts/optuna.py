"""
TITLE: Optuna
AUTHOR: Harrison Myers
DATE: 2025-04-04
DESCRIPTION: Runs an optuna trial on an LSTM model to optimize hyperparameters for DO prediction
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
FIGURE_DIR = os.path.join(PROJECT_DIR, 'outputs', 'figures')
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "optuna")


# Import logging
from utils.logging_setup import setup_logging
logger = setup_logging(log_dir="logs", log_filename="oputna.py")
# logger.info(f"Script initialized: optuna.py")

# Read in data
from scripts.data_preprocessing import river_pipeline, floodplain_pipeline
river_dict = river_pipeline("1")

# Extract data from river dictionary
data_sc_noNA = river_dict['scaled_df']
data_sc_train = river_dict['train_df']
data_sc_test = river_dict['test_df']
riv_data_site_inds = river_dict['split_inds']
X_train = river_dict['X_train']
X_test = river_dict['X_test']
y_train = river_dict['y_train']
y_test = river_dict['y_test']
train_gen = river_dict['train_gen']
test_gen = river_dict['test_gen']
steps_per_epoch = river_dict['steps_per_epoch']
val_steps = river_dict['val_steps']

# Run optuna trial
import optuna
import utils.model_building_training_functions as mbf
import yaml
import random
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras import backend as K
import gc
from keras.callbacks import EarlyStopping


CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')
with open(os.path.join(CONFIG_DIR, 'river_lstm1_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
seed = config['seed']
split_pct = config['split_pct']
n_epochs = config['n_epochs']
activation_dense = config['activation_dense']
stop_patience = config['stop_patience']

def objective(trial):
    try:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
        hidden_units = trial.suggest_categorical('hidden_units', [16, 32, 64])
        window_length = trial.suggest_int('window_length', 5, 365)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr_schedule_patience = trial.suggest_int('lr_schedule_patience', 3, 8)
        
        # Print out selected hyperparameters
        logger.info(f"seed: {seed}")
        logger.info(f"split_pct: {split_pct}")
        logger.info(f"n_epochs: {n_epochs}")
        logger.info(f"activation_dense: {activation_dense}")
        logger.info(f"stop_patience: {stop_patience}")
        logger.info(f"Hidden layers: {hidden_layers}")
        logger.info(f"Hidden units: {hidden_units}")
        logger.info(f"Window length: {window_length}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Dropout: {dropout}")
        logger.info(f"LR schedule patience: {lr_schedule_patience}")
        
        # Build the model
        keras.utils.set_random_seed(seed)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        n_features = len(X_train.columns)
        model = mbf.build_model(hidden_layers, hidden_units, optimizer, window_length, n_features,
                            dropout=dropout, lr=learning_rate, custom_loss_fn=None, batch_norm=False)
        
        # Callbacks
        keras.utils.set_random_seed(seed)
        model_name = f'River_LSTM_'
        earlyStopping = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=1, mode='min')
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=lr_schedule_patience, min_lr=0.0000001, verbose=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(PROJECT_DIR,'models', f'{model_name}.weights.h5'), save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=0)
        callbacks = [earlyStopping, lr_scheduler, checkpoint_callback]
        
        # Train the model
        history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, batch_size=batch_size, 
                            epochs=n_epochs, verbose=1, validation_data=test_gen, 
                            validation_steps=val_steps, callbacks=callbacks)
        logger.info(f'Model ({model_name}) parameters:')
        logger.info(f'Seed: {seed}')
        logger.info(f'Activation: {activation_dense}')
        logger.info(f'Split Percentage: {split_pct}')
        logger.info(f'Hidden LSTM layers: {hidden_layers}')
        logger.info(f'Hidden units: {hidden_units}')
        logger.info(f'Window length: {window_length}')
        logger.info(f'Batch size: {batch_size}')
        logger.info(f'Learning rate: {learning_rate}')
        logger.info(f'Dropout: {dropout}')
        logger.info(f'Epochs: {n_epochs}')
        logger.info(f'Stop patience: {stop_patience}')
        logger.info(f"Learning rate scheduler patience: {lr_schedule_patience}")
        logger.info("Custom loss function: None")
        logger.info(f"Model loss: {history.history['loss']}")
        logger.info(f"Validation loss: {history.history['val_loss']}")
                
        return np.nanmin(history.history['val_loss'])
        
    except Exception as e:
        logger.info(f"Error occurred: {e}")
        
    finally:
        
        # clear GPU memory to ensure efficient usage throughout trial
        logger.info("Clearing GPU Memory and deleting model")
        K.clear_session()
        del model
        gc.collect()
                

db_filepath =OUTPUT_DIR + "/optuna/river_hyperparameter_tuning.db"
storage = optuna.storages.RDBStorage(f'sqlite:///{db_filepath}')

# Enable optuna logging
optuna.logging.set_verbosity(optuna.logging.DEBUG)

study = optuna.create_study(direction='minimize', storage=storage, study_name='river_LSTM_HP_optimization_v3', load_if_exists=True)
study.optimize(objective, n_trials=30)

print("Number of finished trials:", len(study.trials))
print("Best trial:")
best_trial = study.best_trial
logger.info(f"Best Hyperparameters: \n{best_trial}")
print("  Value: ", best_trial.value)
print("  Params: ")
with open(OUTPUT_DIR + "/best_trial_params.txt", "w") as f:   
    for key, value in best_trial.params.items():
        f.write(f"{key}: {value}")
