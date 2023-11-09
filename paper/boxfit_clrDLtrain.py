import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Sequential
from tensorflow.keras.losses import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler
from CLR.clr_callback import CyclicLR

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load datasets
train_features = pd.read_csv("boxfitdata/boxfit_ism_final_trainfeatures.csv")
test_features = pd.read_csv("boxfitdata/boxfit_ism_final_testfeatures.csv")
train_labels = pd.read_csv("boxfitdata/boxfit_ism_final_trainlabels.csv")
test_labels = pd.read_csv("boxfitdata/boxfit_ism_final_testlabels.csv")

# Initialize and fit scalers
scaler_in = StandardScaler()
scaler_out = StandardScaler()
train_features_scaled = scaler_in.fit_transform(train_features)
train_labels_scaled = scaler_out.fit_transform(train_labels)
test_features_scaled = scaler_in.transform(test_features)
test_labels_scaled = scaler_out.transform(test_labels)

# Filepath for saving models
filepath = "boxfitfinal/"


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
    return mae(y_true_no_nan, y_pred_no_nan) * (
        tf.size(y_true, out_type=tf.float32) / non_zero_count
    )


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
    model = Sequential(
        [
            layers.Dense(1000, input_shape=input_shape, activation="softplus"),
            layers.Dense(1000, activation="softplus"),
            layers.Dense(1000, activation="softplus"),
            layers.Dense(117, activation="linear"),
        ]
    )

    model.compile(
        loss=CustomAccuracy(),
        metrics=[masked_metric],
        optimizer=optimizers.Nadam(0.001),
    )
    return model


def train_model(
    model,
    features,
    labels,
    batch_size,
    clr_step_size,
    base_lr,
    max_lr,
    mode,
    epochs,
    filepath,
    Ntrain,
):
    """Train the model with the given dataset and hyperparameters.

    Args:
        model: The neural network model to train.
        features: Input features for training.
        labels: True labels for training.
        batch_size: Size of the batches for training.
        clr_step_size: Step size for cyclic learning rate.
        base_lr: Base learning rate.
        max_lr: Maximum learning rate.
        mode: Mode for cyclic learning rate.
        epochs: Number of epochs to train.
        filepath: Path to save the trained model.
        Ntrain: Number of training samples.
    """
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=mode)
    model.fit(
        features,
        labels,
        validation_split=0.0,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        callbacks=[clr],
    )
    model.save(filepath + "boxfit_ism_stdsc_" + str(Ntrain) + ".h5")


def evaluate_model(model, features, labels, scaler):
    """Evaluate the model on the test dataset and print error metrics.

    Args:
        model: The trained neural network model.
        features: Input features for evaluation.
        labels: True labels for evaluation.
        scaler: Scaler used to inverse transform the predictions.
    """
    test_predictions_scaled = model.predict(features)
    test_predictions_unscaled = scaler.inverse_transform(test_predictions_scaled)
    test_predictions = 10**test_predictions_unscaled
    test_labels_lin = 10**labels
    err = np.abs(test_predictions - test_labels_lin) / test_labels_lin
    err = err.values.flatten()
    print("errors <0.1: " + str(np.mean(err < 0.1)))
    print("errors >0.2: " + str(np.mean(err > 0.2)))
    print("median errors: " + str(np.nanmedian(err)))


# Training loop with different training sizes
trainingsizes = [len(train_features_scaled) // 2**i for i in range(6)]
for Ntrain in trainingsizes:
    dnn_model = build_and_compile_model((train_features_scaled.shape[1],))
    dnn_model.summary()
    train_features_scaled_subset = train_features_scaled[:Ntrain]
    train_labels_scaled_subset = train_labels_scaled[:Ntrain]
    batch_size = 128
    clr_step_size = int(4 * (len(train_features_scaled_subset) / batch_size))
    base_lr = 1e-4
    max_lr = 1e-2
    mode = "triangular2"
    epochs = 200

    train_model(
        dnn_model,
        train_features_scaled_subset,
        train_labels_scaled_subset,
        batch_size,
        clr_step_size,
        base_lr,
        max_lr,
        mode,
        epochs,
        filepath,
        Ntrain,
    )

    evaluate_model(dnn_model, test_features_scaled, test_labels, scaler_out)
