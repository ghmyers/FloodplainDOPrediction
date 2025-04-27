#!/usr/bin/env python3
"""
TITLE: train.py
AUTHOR: Harrison Myers
DATE: 2025-04-27
DESCRIPTION: This script runs preprocessing and training pipelines for river, floodplain, and TL models.
"""
import argparse
import os
import sys
import yaml
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

def _get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Pre-process data and train river, floodplain and TL models.",
    )
    parser.add_argument(
        "--RIV_MODEL_NO",
        required=True,
        type=str,
        metavar="N",
        help="Model number that appears in river_lstm<N>_config.yaml",
    )
    parser.add_argument(
        "--FP_MODEL_NO",
        required=True,
        type=str,
        metavar="N",
        help="Model number that appears in FP_lstm<N>_config.yaml",
    )
    parser.add_argument(
        "--TL_MODEL_NO",
        required=True,
        type=str,
        metavar="N",
        help="Model number that appears in TL_lstm<N>_config.yaml",
    )
    return parser.parse_args()
def main():
    args = _get_cli_args()
    RIV_MODEL_NO = args.RIV_MODEL_NO
    FP_MODEL_NO = args.FP_MODEL_NO
    TL_MODEL_NO = args.TL_MODEL_NO
    
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

    # Import logging
    from src.utils.logging_setup import setup_logging
    logger = setup_logging(log_dir="logs", log_filename="train.py")
    logger.info(f"Script initialized: {os.path.basename(__file__)}")


    # Import data and run preprocessing pipelines
    from scripts.data_preprocessing import river_pipeline, floodplain_pipeline
    river_dict = river_pipeline("3")
    floodplain_dict = floodplain_pipeline("4")

    # Extract necessary features from dictionaries

    ###### FLOODPLAIN DATA ##############################################
    # Get master (i.e., concatenated across all sites) data 
    X_train_FP = floodplain_dict['master']['X_train_fp']
    y_train_FP = floodplain_dict['master']['y_train_fp']
    X_test_FP= floodplain_dict['master']['X_test_fp']
    y_test_FP = floodplain_dict['master']['y_test_fp']
    train_gen_FP = floodplain_dict['master']['train_gen_fp']
    test_gen_FP = floodplain_dict['master']['test_gen_fp']
    steps_per_epoch_FP = floodplain_dict['master']['steps_per_epoch_fp']
    val_steps_FP = floodplain_dict['master']['val_steps_fp']
    X_FP_rs = floodplain_dict['master']['X_master_rs']
    y_FP_rs = floodplain_dict['master']['y_master_rs']

    # Get site level data
    # OC2
    OC2_noNA = floodplain_dict['OC2']['OC2_noNA']
    X_OC2 = floodplain_dict['OC2']['X_OC2']
    y_OC2 = floodplain_dict['OC2']['y_OC2']
    OC2_start_ind = floodplain_dict['OC2']['OC2_start']
    OC2_end_ind = floodplain_dict['OC2']['OC2_end']
    # LF1
    LF1_noNA = floodplain_dict['LF1']['LF1_noNA']
    X_LF1 = floodplain_dict['LF1']['X_LF1']
    y_LF1 = floodplain_dict['LF1']['y_LF1']
    LF1_start_ind = floodplain_dict['LF1']['LF1_start']
    LF1_end_ind = floodplain_dict['LF1']['LF1_end']
    # LF2
    LF2_noNA = floodplain_dict['LF2']['LF2_noNA']
    X_LF2 = floodplain_dict['LF2']['X_LF2']
    y_LF2 = floodplain_dict['LF2']['y_LF2']
    LF2_start_ind = floodplain_dict['LF2']['LF2_start']
    LF2_end_ind = floodplain_dict['LF2']['LF2_end']
    # OC4
    OC4_noNA = floodplain_dict['OC4']['OC4_noNA']
    X_OC4 = floodplain_dict['OC4']['X_OC4']
    y_OC4 = floodplain_dict['OC4']['y_OC4']
    OC4_start_ind = floodplain_dict['OC4']['OC4_start']
    OC4_end_ind = floodplain_dict['OC4']['OC4_end']
    # LF3
    LF3_noNA = floodplain_dict['LF3']['LF3_noNA']
    X_LF3 = floodplain_dict['LF3']['X_LF3']
    y_LF3 = floodplain_dict['LF3']['y_LF3']
    LF3_start_ind = floodplain_dict['LF3']['LF3_start']
    LF3_end_ind = floodplain_dict['LF3']['LF3_end']
    # OC1
    OC1_noNA = floodplain_dict['OC1']['OC1_noNA']
    X_OC1 = floodplain_dict['OC1']['X_OC1']
    y_OC1 = floodplain_dict['OC1']['y_OC1'] 
    OC1_start_ind = floodplain_dict['OC1']['OC1_start']
    OC1_end_ind = floodplain_dict['OC1']['OC1_end']
    # OC3
    OC3_noNA = floodplain_dict['OC3']['OC3_noNA']
    X_OC3 = floodplain_dict['OC3']['X_OC3']
    y_OC3 = floodplain_dict['OC3']['y_OC3']
    OC3_start_ind = floodplain_dict['OC3']['OC3_start']
    OC3_end_ind = floodplain_dict['OC3']['OC3_end']

    ###### RIVER DATA ##############################################
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

    # Get model hyperparameters
    import yaml
    CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')
    with open(os.path.join(CONFIG_DIR, f'river_lstm{RIV_MODEL_NO}_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        
    seed = config['seed']
    split_pct = config['split_pct']
    window_length = config['window_length']
    hidden_layers = config['hidden_layers']
    hidden_units = config['hidden_units']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    activation_dense = config['activation_dense']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    stop_patience = config['stop_patience']
    lr_schedule_patience = config['lr_schedule_patience']

    ##### Train the river LSTM model ################################################
    import keras
    from src.utils import model_building_training_functions as mbf

    # Build the model
    keras.utils.set_random_seed(seed)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    n_features = len(X_train.columns)
    # n_features = 27
    model = mbf.build_model(hidden_layers, hidden_units, optimizer, window_length, n_features,
                        dropout=dropout, lr=learning_rate, custom_loss_fn=None, batch_norm=False)

    # Train the model

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    keras.utils.set_random_seed(seed)
    model_name = 'River_LSTM'
    earlyStopping = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=1, mode='min')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=lr_schedule_patience, min_lr=0.0000001, verbose=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(PROJECT_DIR,'models', f'{model_name}_{RIV_MODEL_NO}.weights.h5'), save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=0)

    callbacks = [earlyStopping, lr_scheduler, checkpoint_callback]
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

    # Plot history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4), dpi=400)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('River LSTM Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # plt.savefig(os.path.join(FIGURE_DIR, 'river_LSTM_loss.png'))
    plt.show()

    ##### Train TL LSTM ############################################################
    with open(os.path.join(CONFIG_DIR, f"TL_lstm{TL_MODEL_NO}_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Set hyperparameters from config file
    seed = config['seed']
    split_pct = config['split_pct']
    window_length = config['window_length']
    hidden_layers = config['hidden_layers']
    hidden_units = config['hidden_units']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    activation_dense = config['activation_dense']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    stop_patience = config['stop_patience']
    lr_schedule_patience = config['lr_schedule_patience']


    keras.utils.set_random_seed(seed)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    n_features = len(X_train_FP.columns)
    model_FT = mbf.build_model(hidden_layers, hidden_units, optimizer, window_length, n_features,
                        dropout=dropout, lr=learning_rate, custom_loss_fn=None, batch_norm=False)

    model_FT.load_weights(PROJECT_DIR + '/models/old/River_LSTM.weights.h5')

    model_name_FT = "TL_LSTM_{TL_MODEL_NO}"

    keras.utils.set_random_seed(seed)
    earlyStopping_FT = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=1, mode='min')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=lr_schedule_patience, min_lr=0.0000001, verbose=1)
    checkpoint_callback_FT = tf.keras.callbacks.ModelCheckpoint(filepath=PROJECT_DIR + f'/models/{model_name_FT}.weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=0)

    callbacks_FT = [earlyStopping_FT, lr_scheduler, checkpoint_callback_FT]
    history_FT = model_FT.fit(train_gen_FP, steps_per_epoch=steps_per_epoch_FP, batch_size=batch_size, 
                        epochs=n_epochs, verbose=1, validation_data=test_gen_FP, 
                        validation_steps=val_steps_FP, callbacks=callbacks_FT)

    logger.info(f'Model ({model_name_FT}) parameters:')
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
    logger.info(f"Model loss: {history_FT.history['loss']}")
    logger.info(f"Validation loss: {history_FT.history['val_loss']}")

    # Plot history and save
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(history_FT.history['loss'])
    plt.plot(history_FT.history['val_loss'])
    plt.title('TL-LSTM Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    # plt.savefig(FIGURE_DIR + '/TL_LSTM_v4_oldFeatures_loss.png', dpi=300)
    plt.show()

    ##### Train the FLoodplain LSTM model ###########################################
    # Get hyperparameters
    with open(os.path.join(CONFIG_DIR, f"FP_lstm{FP_MODEL_NO}_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Set hyperparameters from config file
    seed = config['seed']
    split_pct = config['split_pct']
    window_length = config['window_length']
    hidden_layers = config['hidden_layers']
    hidden_units = config['hidden_units']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    activation_dense = config['activation_dense']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    stop_patience = config['stop_patience']
    lr_schedule_patience = config['lr_schedule_patience']

    # Instantiate model
    keras.utils.set_random_seed(seed)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    n_features = len(X_train_FP.columns)
    model_FP = mbf.build_model(hidden_layers, hidden_units, optimizer, window_length, n_features,
                        dropout=dropout, lr=learning_rate, custom_loss_fn=None, batch_norm=False)

    model_name = f"FP_LSTM_{FP_MODEL_NO}"
    keras.utils.set_random_seed(seed)
    earlyStopping_FP = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=1, mode='min')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=lr_schedule_patience, min_lr=0.0000001, verbose=1)
    checkpoint_callback_FP = tf.keras.callbacks.ModelCheckpoint(filepath=PROJECT_DIR + f'/models/{model_name}.weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=0)

    callbacks_FP = [earlyStopping_FP, lr_scheduler, checkpoint_callback_FP]
    history_FP = model_FP.fit(train_gen_FP, steps_per_epoch=steps_per_epoch_FP, batch_size=batch_size, 
                        epochs=n_epochs, verbose=1, validation_data=test_gen_FP, 
                        validation_steps=val_steps_FP, callbacks=callbacks_FP)

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
    logger.info(f"Model loss: {history_FP.history['loss']}")
    logger.info(f"Validation loss: {history_FP.history['val_loss']}")

    # Save model history
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(history_FP.history['loss'])
    plt.plot(history_FP.history['val_loss'])
    plt.title('FP-LSTM Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    # plt.savefig(FIGURE_DIR + '/FP_LSTM_v4_loss.png', dpi=300)
    plt.show()
    
if __name__ == "__main__":
    main()