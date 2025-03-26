import tensorflow as tf
from keras import layers, metrics
import keras
import numpy as np

#%% Custom loss functions
def smape_loss(y_true, y_pred):
    # Don't update loss if value is nan
    # mask = tf.not_equal(y_true, 0)
    # y_true = tf.boolean_mask(y_true, mask)
    # y_pred = tf.boolean_mask(y_pred, mask)
    
    denominator = tf.abs(y_true) + tf.abs(y_pred) + tf.keras.backend.epsilon() # Adding epsilon to avoid division by zero
    diff = tf.abs(y_true - y_pred)
    numerator = tf.abs(diff)
    smape = 2.0 * tf.reduce_mean(numerator / denominator) * 100
    return smape
    
def nse_loss(y_true, y_pred):
    # Don't update loss if value is nan
    # mask = tf.not_equal(y_true, 0)
    # y_true = tf.boolean_mask(y_true, mask)
    # y_pred = tf.boolean_mask(y_pred, mask)
    
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    nse_loss = numerator / (denominator + tf.keras.backend.epsilon()) # Adding epsilon to avoid division by zero
    return nse_loss

def nse_loss_low(y_true, y_pred):
    # Don't update loss if value is nan
    # mask = tf.not_equal(y_true, 0)
    # y_true = tf.boolean_mask(y_true, mask)
    # y_pred = tf.boolean_mask(y_pred, mask)
    
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    nse_loss = numerator / (denominator + tf.keras.backend.epsilon()) # Adding epsilon to avoid division by zero
    
    # Punish loss for values less than 3 twice as much
    condition = tf.less_equal(y_true, 3)  # Create condition for y_true <= 3
    multiplier = tf.where(condition, 2.0, 1.0)  # Multiply loss by 2 if y_true <= 3
    
    # Apply the multiplier to the loss
    nse_loss = nse_loss * tf.reduce_mean(multiplier)  # Averaging over batch
    
    return nse_loss

def kge_loss(y_true, y_pred):
    # Don't update loss if value is nan
    # mask = tf.not_equal(y_true, mask_val)
    # y_true = tf.boolean_mask(y_true, mask)
    # y_pred = tf.boolean_mask(y_pred, mask)
    
    # Calculate mean and standard deviation
    mean_obs = tf.reduce_mean(y_true)
    mean_sim = tf.reduce_mean(y_pred)
    std_obs = tf.math.reduce_std(y_true)
    std_sim = tf.math.reduce_std(y_pred)
    
    # Calculate Pearson correlation coefficient
    r_num = tf.reduce_sum((y_true - mean_obs) * (y_pred - mean_sim))
    r_den = tf.sqrt(tf.reduce_sum(tf.square(y_true - mean_obs)) * tf.reduce_sum(tf.square(y_pred - mean_sim)))
    r = r_num / (r_den + tf.keras.backend.epsilon())
    
    # Calculate alpha and beta
    alpha = std_sim / (std_obs + tf.keras.backend.epsilon())
    beta = mean_sim / (mean_obs + tf.keras.backend.epsilon())
    
    # Calculate KGE loss
    kge_loss = tf.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_loss

def kge_loss(y_true, y_pred):
    # Don't update loss if value is nan
    # mask = tf.not_equal(y_true, mask_val)
    # y_true = tf.boolean_mask(y_true, mask)
    # y_pred = tf.boolean_mask(y_pred, mask)
    
    # Calculate mean and standard deviation
    mean_obs = tf.reduce_mean(y_true)
    mean_sim = tf.reduce_mean(y_pred)
    std_obs = tf.math.reduce_std(y_true)
    std_sim = tf.math.reduce_std(y_pred)
    
    # Calculate Pearson correlation coefficient
    r_num = tf.reduce_sum((y_true - mean_obs) * (y_pred - mean_sim))
    r_den = tf.sqrt(tf.reduce_sum(tf.square(y_true - mean_obs)) * tf.reduce_sum(tf.square(y_pred - mean_sim)))
    r = r_num / (r_den + tf.keras.backend.epsilon())
    
    # Calculate alpha and beta
    alpha = std_sim / (std_obs + tf.keras.backend.epsilon())
    beta = mean_sim / (mean_obs + tf.keras.backend.epsilon())
    
    # Calculate KGE loss
    kge_loss = tf.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    
    # Punish loss for values less than 3 twice as much
    condition = tf.less_equal(y_true, 3)  # Create condition for y_true <= 3
    multiplier = tf.where(condition, 2.0, 1.0)  # Multiply loss by 2 if y_true <= 3
    
    # Apply the multiplier to the loss
    kge_loss = kge_loss * tf.reduce_mean(multiplier)  # Averaging over batch
    
    return kge_loss


def mse_loss_low(y_true, y_pred):
    # Compute the MSE
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Create a condition for y_true <= 3
    condition = tf.less_equal(y_true, 3)
    
    # Apply a multiplier of 2 if y_true <= 3, otherwise 1
    multiplier = tf.where(condition, 2.0, 1.0)
    
    # Apply the multiplier to the MSE loss
    mse_loss = mse_loss * tf.reduce_mean(multiplier)  # Adjust loss by the mean multiplier
    
    return mse_loss

#%% Function to build model
def build_model(hidden_layers, hidden_units, optimizer, window_length, n_features, lr, dropout=0, custom_loss_fn=None, mask=False, batch_norm=False, mask_val=-1, activation_dense="relu"):
    loss_fn = custom_loss_fn if custom_loss_fn else 'mean_squared_error'
    optimizer = optimizer
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_length, n_features)))
    if mask:
        model.add(layers.Masking(mask_value=mask_val, input_shape=(window_length, n_features)))
    for i in range(hidden_layers):
        if i != hidden_layers - 1:
            model.add(layers.LSTM(units=hidden_units, return_sequences=True))
            model.add(layers.Dropout(dropout))
            if batch_norm:
                model.add(layers.BatchNormalization())
        else:
            model.add(layers.LSTM(units=hidden_units))
            model.add(layers.Dropout(dropout))
            if batch_norm: 
                model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=1, activation=activation_dense))
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["mean_absolute_error", metrics.RootMeanSquaredError()])
    model.summary()
              
    return model

#%% Learning rate scheduler functions
# Exponential decay
def exponential_decay_scheduler(initial_lr, decay_rate, decay_steps):
    """
    Decays learning rate to decay_rate*initial_lr after [decay_steps] epochs
    """
    def scheduler(epoch):
        return float(initial_lr * decay_rate ** (epoch / decay_steps))
    return scheduler

# Step decay
def step_decay_scheduler(initial_lr, drop_rate, epochs_drop):
    """
    Decays learning rate by drop_rate every epochs_drop epochs
    """
    def scheduler(epoch):
        return float(initial_lr * (drop_rate ** (epoch // epochs_drop)))
    return scheduler

# Polynomial decay
def polynomial_decay_scheduler(initial_lr, end_lr, decay_steps, power):
    """
    Decays learning rate by a power polynomial until decay_steps epoch has been reached, then keeps lr constant at end_lr
    """
    def scheduler(epoch):
        while epoch <= decay_steps:
            decay_factor = (1 - (epoch / decay_steps)) ** power
            return float((initial_lr - end_lr) * decay_factor + end_lr)
        else: # avoid imaginary numbers when epoch > decay_steps
            return float(end_lr)
    return scheduler

# Cosine decay
def cosine_decay_scheduler(initial_lr, decay_steps):
    """
    Decays learning rate by a cosine function until lr reaches 0 at decay_steps epochs, then symetrically increases lr by the same function
    """
    def scheduler(epoch):
        return float(initial_lr * 0.5 * (1 + tf.cos(np.pi * epoch / decay_steps)))
    return scheduler   

