# U.S. Electric Production Forecasting: Comparative Study

# 📦 Install Required Libraries
# !pip install prophet xgboost pytorch-lightning pytorch-forecasting --quiet

# 📚 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet

# Deep learning imports
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, NBeats, DeepAR, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder, GroupNormalizer
from pytorch_forecasting.metrics import SMAPE



# 📥 Load Dataset
df = pd.read_csv("Electric_Production.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
train = df.iloc[:-12]
test = df.iloc[-12:]
y_test = test['IPG2211A2N'].values


# 📏 Accuracy Function
def forecast_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

results = {}


# 🔢 Impermanence Forecast (Seasonal + Hybrid AR)
monthly_pct_changes = [0] * 12
monthly_counts = [0] * 12
for i in range(1, len(train)):
    prev_month = train.index[i - 1].month
    pct_change = (train['IPG2211A2N'].iloc[i] - train['IPG2211A2N'].iloc[i - 1]) / train['IPG2211A2N'].iloc[i - 1]
    monthly_pct_changes[prev_month - 1] += pct_change
    monthly_counts[prev_month - 1] += 1
monthly_avg_deltas = [total / count if count > 0 else 0 for total, count in zip(monthly_pct_changes, monthly_counts)]

seasonal_forecast = [train['IPG2211A2N'].iloc[-1]]
last_month = train.index[-1].month
for i in range(12):
    delta = monthly_avg_deltas[(last_month + i - 1) % 12]
    seasonal_forecast.append(seasonal_forecast[-1] * (1 + delta))
seasonal_forecast = seasonal_forecast[1:]

# Calculate month-to-month average relative changes over the years
train_monthly = train.copy()
train_monthly['Month'] = train_monthly.index.month

# Compute average relative change from month to next month
monthly_pct_changes = []
for m in range(1, 13):
    this_month = train_monthly[train_monthly['Month'] == m]['IPG2211A2N'].values
    next_month = train_monthly[train_monthly['Month'] == ((m % 12) + 1)]['IPG2211A2N'].values
    
    # Align lengths and compute percentage change to next month
    min_length = min(len(this_month), len(next_month))
    if min_length > 0:  # Ensure there is data to compute
        this_month = this_month[:min_length]
        next_month = next_month[:min_length]
        pct_change = (next_month - this_month) / this_month
        monthly_pct_changes.append(pct_change.mean())
    else:
        monthly_pct_changes.append(0)  # Default to 0 if no data is available

# Forecast using seasonal deltas
seasonal_impermanence_forecast = [train['IPG2211A2N'].iloc[-1]]
last_month = train.index[-1].month
for i in range(12):
    delta = monthly_pct_changes[(last_month + i - 1) % 12]
    next_val = seasonal_impermanence_forecast[-1] * (1 + delta)
    seasonal_impermanence_forecast.append(next_val)

# Remove initial value
seasonal_impermanence_forecast = seasonal_impermanence_forecast[1:]

# Evaluate performance
results['Seasonal Impermanence'] = forecast_accuracy(y_test, seasonal_impermanence_forecast)


# Hybrid AR correction
lookback = 24
train_tail = train[-lookback:]
train_tail_forecast = [train_tail['IPG2211A2N'].iloc[0]]
for i in range(1, lookback):
    month = train_tail.index[i - 1].month
    delta = monthly_avg_deltas[month - 1]
    train_tail_forecast.append(train_tail_forecast[-1] * (1 + delta))
residuals = train_tail['IPG2211A2N'].values[1:] - np.array(train_tail_forecast[1:])
ar_model = AutoReg(residuals, lags=6).fit()
ar_residuals = ar_model.predict(start=len(residuals), end=len(residuals)+11)
hybrid_forecast = np.array(seasonal_forecast) + ar_residuals
results['Hybrid Seasonal + AR'] = forecast_accuracy(y_test, hybrid_forecast)

# 📉 Classical Models
ar = AutoReg(train['IPG2211A2N'], lags=12).fit().predict(start=len(train), end=len(train)+11)
results['AR'] = forecast_accuracy(y_test, ar)

arima = ARIMA(train['IPG2211A2N'], order=(5,1,0)).fit().forecast(steps=12)
results['ARIMA'] = forecast_accuracy(y_test, arima)

sarima = SARIMAX(train['IPG2211A2N'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False).forecast(steps=12)
results['SARIMA'] = forecast_accuracy(y_test, sarima)

es = ExponentialSmoothing(train['IPG2211A2N'], trend='add', seasonal='add', seasonal_periods=12).fit().forecast(12)
results['Exponential Smoothing'] = forecast_accuracy(y_test, es)

# 🚀 Prophet
prophet_train = train.reset_index().rename(columns={'DATE': 'ds', 'IPG2211A2N': 'y'})
prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(prophet_train)
future = prophet_model.make_future_dataframe(periods=12, freq='MS')
prophet_pred = prophet_model.predict(future).tail(12)['yhat'].values
results['Prophet'] = forecast_accuracy(y_test, prophet_pred)

# 🤖 XGBoost
def create_lagged_features(series, lags=12):
    df_lags = pd.concat([series.shift(i) for i in range(1, lags + 1)], axis=1)
    df_lags.columns = [f'lag_{i}' for i in range(1, lags + 1)]
    return df_lags

X_train = create_lagged_features(train['IPG2211A2N']).dropna()
y_train = train['IPG2211A2N'].iloc[12:]
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

last_values = list(train['IPG2211A2N'].values[-12:])
xgb_pred = []
for _ in range(12):
    x = np.array(last_values[-12:]).reshape(1, -1)
    pred = model.predict(x)[0]
    xgb_pred.append(pred)
    last_values.append(pred)
results['XGBoost'] = forecast_accuracy(y_test, xgb_pred)

# 📥 Load Dataset
df = pd.read_csv("Electric_Production.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df['time_idx'] = (df['DATE'].dt.year - df['DATE'].dt.year.min()) * 12 + df['DATE'].dt.month
df['group'] = "electricity"

# Train/test split
train = df.iloc[:-12].copy()
test = df.iloc[-12:].copy()

# Deep Learning Dataset Prep
max_encoder_length = 36
max_prediction_length = 12

# Define the TimeSeriesDataSet with only the target as input
training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="IPG2211A2N",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=[],  # No known reals
    time_varying_unknown_reals=["IPG2211A2N"],  # Only the target is unknown
    target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),  # Use continuous normalizer
    add_relative_time_idx=False,  # Set to False to avoid conflicts
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation set
testing = TimeSeriesDataSet.from_dataset(training, df, stop_randomization=True, predict=True)

# Create dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=32)
test_dataloader = testing.to_dataloader(train=False, batch_size=32)

# Define trainer
# trainer = pl.Trainer(
#     max_epochs=30,
#     gradient_clip_val=0.1,
#     enable_model_summary=True,
#     accelerator="cpu",  # Use "cpu" instead of "gpu"
#     devices=1           # Use 1 CPU device
# )

# # Initialize DeepAR model
# deepar = DeepAR.from_dataset(training, learning_rate=0.01)

# # Train the model
# trainer.fit(deepar, train_dataloader)

# # Make predictions
# preds = deepar.predict(test_dataloader, mode="prediction").detach().numpy().flatten()
# results["DeepAR"] = forecast_accuracy(test["IPG2211A2N"].values, preds)

# N-BEATS
# nbeats = NBeats.from_dataset(training, learning_rate=0.01)
# trainer.fit(nbeats, train_dataloader)
# preds = nbeats.predict(test_dataloader, mode="prediction").detach().numpy().flatten()
# results["N-BEATS"] = forecast_accuracy(test["IPG2211A2N"].values, preds)

# TFT
# tft = TemporalFusionTransformer.from_dataset(training, learning_rate=0.01)
# trainer.fit(tft, train_dataloader)
# preds = tft.predict(test_dataloader, mode="prediction").detach().numpy().flatten()
# results["TFT"] = forecast_accuracy(test["IPG2211A2N"].values, preds)

# 📊 Final Results
result_df = pd.DataFrame(results).T
print(result_df)

# You can also visualize forecasts vs actual here.
