"""
forecast.py - Reproduce the forecasting step of paper
"Snow Water Equivalent Forecasting in Sub-Arctic and Arctic Regions: Efficient Recurrent Neural Networks Approach"
Copyright (C) 2025  Miika Malin
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from t2v.Time2Vec.layers import Time2Vec


def nse(y_true, y_pred):
    """Calculate Nash–Sutcliffe model efficiency coefficient"""
    nse = (1
           - (np.sum((y_true - y_pred)**2)
              / np.sum((y_true-np.mean(y_true))**2)))
    return nse


def de_normalize(y_norm, min_swe=0, max_swe=260):
    """Denormalize forecast back to original scale"""
    denormalized = (y_norm * (max_swe - min_swe)) + min_swe
    return denormalized


def plot_forecast(ax, real, forecast, title):
    ax.plot(real, label="Real", c="green")
    ax.plot(forecast, label="Forecast", linestyle="--", marker='o', c="blue")
    ax.set_ylabel("SWE")
    ax.legend(loc="upper left")
    ax.set_title(title)


# List of all stations
STATIONS = ["lohja", "vaala", "inari"]


# ---------- Data loading ----------
print("Loading in preprocessed data...")
data = {}
for station in STATIONS:
    data[f"lightweight_input_{station}"] = np.load(
        f"data/lightweight/{station}_x_test.npy")
    data[f"lightweight_true_swe_{station}"] = np.load(
        f"data/lightweight/{station}_y_test.npy")
    data[f"heavy_input_{station}"] = [
            np.load(f"data/heavy/{station}_x_ts_test.npy"),
            np.load(f"data/heavy/{station}_x_test.npy")]
    data[f"heavy_true_swe_{station}"] = np.load(
        f"data/heavy/{station}_y_test.npy")


# ---------- Define models ----------
print("Initializing models...")
# Define lightweight model
lightweight_model = tf.keras.models.Sequential()
lightweight_model.add(tf.keras.layers.GRU(8, input_shape=(12, 3)))
lightweight_model.add(tf.keras.layers.Dense(1))
lightweight_model.compile()
# Define heavy model
input_timestep = tf.keras.layers.Input((180, 1))
input_features = tf.keras.layers.Input((180, 3))
x_timestep = Time2Vec(kernel_size=1)(input_timestep)
masked = tf.keras.layers.Masking(mask_value=-1)(input_features)
concat = tf.concat([x_timestep, masked], axis=2)
x_gru = tf.keras.layers.GRU(128)(concat)
output = tf.keras.layers.Dense(1)(x_gru)
heavy_model = tf.keras.models.Model([input_timestep, input_features], output)


# ---------- Predict with lightweight model ----------
print("Forecasting with lightweight model...")
for train_station in STATIONS:
    print(f"\nUsing model trained in {train_station} in forecast")
    fig, axs = plt.subplots(3, 1)
    # Load in weights which has been trained with current station
    weight_path = f"weights/lightweight/{train_station}/"
    weights = tf.train.latest_checkpoint(weight_path)
    lightweight_model.load_weights(weights)
    # Predict all stations
    i = 0
    print("NSE-values:")
    for frcst_station, i in zip(STATIONS, range(len(STATIONS))):
        model_input = data[f"lightweight_input_{frcst_station}"]
        real_swe = data[f"lightweight_true_swe_{frcst_station}"]
        forecast = lightweight_model.predict(model_input)
        # Scale forecast back to original scale
        forecast = de_normalize(forecast)
        # Cut out negative values from the GRU forecast
        forecast = forecast.clip(min=0).flatten()
        tmp_nse = round(nse(real_swe, forecast), 2)
        print(f"{frcst_station}: {tmp_nse}")
        # Plot the result
        plot_forecast(axs[i], real_swe, forecast,
                      f"{frcst_station.title()}, NSE={tmp_nse}")
        plt.suptitle(
            f"Forecasting results with lightweight model trained "
            + f"in {train_station.title()}")
    plt.show()


# ---------- Predict with heavy model ----------
print("\n\nForecasting with heavy model...")
# Load in weights
weight_path = f"weights/heavy_model/"
weights = tf.train.latest_checkpoint(weight_path)
heavy_model.load_weights(weights)
# Predict all stations
fig, axs = plt.subplots(3, 1)
print("NSE-values:")
for station, i in zip(STATIONS, range(len(STATIONS))):
    model_input = data[f"heavy_input_{station}"]
    real_swe = data[f"heavy_true_swe_{station}"]
    forecast = heavy_model.predict(model_input)
    # Scale forecast back to original scale
    forecast = de_normalize(forecast)
    # Cut out negative values from the GRU forecast
    forecast = forecast.clip(min=0).flatten()
    tmp_nse = round(nse(real_swe, forecast), 2)
    print(f"{station}: {tmp_nse}")
    plot_forecast(axs[i], real_swe, forecast,
                  f"{station.title()}, NSE={tmp_nse}")
    plt.suptitle(
        f"Forecasting results with heavy model trained in all stations")
# Plot the results
plt.show()
