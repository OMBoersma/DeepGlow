import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Sequential, callbacks
from tensorflow.keras.losses import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler
from CLR.clr_callback import CyclicLR
import matplotlib

# Set CUDA device and matplotlib backend
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
matplotlib.use('Agg')

# Load datasets
train_features = pd.read_csv('boxfitdata/boxfit_wind_final_trainfeatures.csv')
test_features = pd.read_csv('boxfitdata/boxfit_wind_final_testfeatures.csv')
train_labels = pd.read_csv('boxfitdata/boxfit_wind_final_trainlabels.csv')
test_labels = pd.read_csv('boxfitdata/boxfit_wind_final_testlabels.csv')

# Initialize and fit scalers
scaler_in = StandardScaler()
scaler_out = StandardScaler()
train_features_scaled = scaler_in.fit_transform(train_features)
train_labels_scaled = scaler_out.fit_transform(train_labels)
test_features_scaled = scaler_in.transform(test_features)
test_labels_scaled = scaler_out.transform(test_labels)

# Filepath for saving models
filepath = 'boxfitfinal/'
filepath_intermediate = 'boxfitfinal/model-wind-stdsc-{epoch:02d}.hdf5'

def masked_metric(y_true, y_pred):
    """Calculate the masked mean absolute error metric.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    
    Returns:
        The masked mean absolute error.
    """
    mae = MeanAbsoluteError()
    mask = ~tf.math.is_nan(y_true)
    y_true_no_nan = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred_no_nan = tf.where(mask, y_pred, tf.zeros_like(y_pred))
    non_zero_count = tf.reduce_sum(tf.cast(mask, tf.float32))
    return mae(y_true_no_nan, y_pred_no_nan) * (tf.size(y_true, out_type=tf.float32) / non_zero_count)

class CustomAccuracy(losses.Loss):
    """Custom accuracy class that calculates the masked mean absolute error."""
    def call(self, y_true, y_pred):
        return masked_metric(y_true, y_pred)

def build_and_compile_model(input_shape):
    """Build and compile the neural network model.
    
    Args:
        input_shape: Shape of the input data.
    
    Returns:
        A compiled Keras model.
    """
    model = Sequential([
        layers.Dense(1000, input_shape=input_shape, activation='softplus'),
        layers.Dense(1000, activation='softplus'),
        layers.Dense(1000, activation='softplus'),
        layers.Dense(117, activation='linear')
    ])

    model.compile(loss=CustomAccuracy(), metrics=[masked_metric],
                  optimizer=optimizers.Nadam(0.001))
    return model

# Build and compile the model
dnn_model = build_and_compile_model((train_features_scaled.shape[1],))
dnn_model.summary()

# Set up training parameters
batch_size = 128
clr_step_size = int(4 * (len(train_features_scaled) / batch_size))
base_lr = 1e-4
max_lr = 1e-2
mode = 'triangular2'

# Set up callbacks
clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=mode)
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath_intermediate, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=20)

# Train the model
history = dnn_model.fit(train_features_scaled, train_labels_scaled,
                        validation_split=0.0, batch_size=batch_size, verbose=1, epochs=2000, callbacks=[clr, model_checkpoint_callback])

# Save the final model
dnn_model.save(filepath + 'boxfit_wind_final_stdsc.h5')

# Evaluate the model
test_predictions_scaled = dnn_model.predict(test_features_scaled)
test_predictions_unscaled = scaler_out.inverse_transform(test_predictions_scaled)
test_predictions = 10 ** test_predictions_unscaled
test_labels = 10 ** test_labels
err = np.abs(test_predictions - test_labels) / test_labels
err = err.values.flatten()
print('errors <0.1: ' + str(np.mean(err < 0.1)))
print('errors >0.2: ' + str(np.mean(err > 0.2)))
print('median errors: ' + str(np.nanmedian(err)))

