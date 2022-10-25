from CLR.clr_callback import CyclicLR
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.losses import MeanAbsoluteError
from keras import layers
import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

matplotlib.use('Agg')

train_features = pd.read_csv(
    'boxfitdata/boxfit_wind_final_trainfeatures.csv')
test_features = pd.read_csv(
    'boxfitdata/boxfit_wind_final_testfeatures.csv')
train_labels = pd.read_csv(
    'boxfitdata/boxfit_wind_final_trainlabels.csv')
test_labels = pd.read_csv(
    'boxfitdata/boxfit_wind_final_testlabels.csv')

scaler_in = StandardScaler()
scaler_out = StandardScaler()

train_features_scaled = scaler_in.fit_transform(train_features)
test_features_scaled = scaler_in.transform(test_features)


train_labels_scaled = scaler_out.fit_transform(train_labels)
test_labels_scaled = scaler_out.transform(test_labels)

batch_size = 128
clr_step_size = int(4 * (len(train_features_scaled)/batch_size))
base_lr = 1e-4
max_lr = 1e-2
mode = 'triangular2'
clr = CyclicLR(base_lr=base_lr, max_lr=max_lr,
               step_size=clr_step_size, mode=mode)
filepath = 'boxfitfinal/'
filepath_intermediate = 'boxfitfinal/model-wind-stdsc-{epoch:02d}.hdf5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath_intermediate, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=20)


def masked_metric(y_true, y_pred):
    mae = MeanAbsoluteError()
    y_true_no_nan = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_true), y_true)
    y_pred_no_nan = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_pred), y_pred)
    non_zero_cor = tf.cast(tf.size(y_true_no_nan, out_type=tf.int32), dtype=tf.float32) / \
        tf.math.count_nonzero(y_true_no_nan, dtype=tf.float32)
    return mae(y_true_no_nan, y_pred_no_nan)*non_zero_cor


class CustomAccuracy(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        mae = MeanAbsoluteError()
        y_true_no_nan = tf.where(tf.math.is_nan(
            y_true), tf.zeros_like(y_true), y_true)
        y_pred_no_nan = tf.where(tf.math.is_nan(
            y_true), tf.zeros_like(y_pred), y_pred)
        non_zero_cor = tf.cast(tf.size(y_true_no_nan, out_type=tf.int32), dtype=tf.float32) / \
            tf.math.count_nonzero(y_true_no_nan, dtype=tf.float32)
        return mae(y_true_no_nan, y_pred_no_nan)*non_zero_cor


def build_and_compile_model():
    model = keras.Sequential([
        layers.Dense(
            1000, input_dim=train_features_scaled.shape[1], activation='softplus'),
        layers.Dense(1000, activation='softplus'),
        layers.Dense(1000, activation='softplus'),
        layers.Dense(117, activation='linear')
    ])

    model.compile(loss=CustomAccuracy(), metrics=[masked_metric],
                  optimizer=keras.optimizers.Nadam(0.001))
    return model


dnn_model = build_and_compile_model()
dnn_model.summary()

history = dnn_model.fit(train_features_scaled, train_labels_scaled,
                        validation_split=0.0, batch_size=batch_size, verbose=1, epochs=2000, callbacks=[clr, model_checkpoint_callback])

dnn_model.save(filepath+'boxfit_wind_final_stdsc.h5')


test_predictions_scaled = dnn_model.predict(test_features_scaled)
test_predictions_unscaled = scaler_out.inverse_transform(
    test_predictions_scaled)
test_predictions = 10**test_predictions_unscaled
test_labels = 10**test_labels
err = np.abs(test_predictions-test_labels)/test_predictions
err = err.values.flatten()
print('errors <0.1: '+str(len(err[err < 0.1])/len(err)))
print('errors >0.2: '+str(len(err[err > 0.2])/len(err)))
print('median errors: '+str(np.nanmedian(err)))
