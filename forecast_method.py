import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Simulate time series data
np.random.seed(42)
months = 36
trend = np.linspace(100, 200, months)
seasonality = 20 * np.sin(np.linspace(0, 3 * np.pi, months))
promotion_effect = np.random.choice([0, 10, 20], size=months, p=[0.8, 0.1, 0.1])
noise = np.random.normal(0, 10, months)
demand = trend + seasonality + promotion_effect + noise
dates = pd.date_range(start='2022-01-01', periods=months, freq='M')

# 2. Create dataframe
df = pd.DataFrame({'Date': dates, 'Demand': demand})
df.set_index('Date', inplace=True)

# 3. Add external context features
df['ExternalContext'] = 0
df.loc['2023-03-31':'2023-08-31', 'ExternalContext'] = -30  # Simulated COVID dip
df['CalendarEffect'] = np.tile([0, 5, -5], months // 3)[:months]
df['PromotionFlag'] = np.random.choice([0, 1], size=months, p=[0.8, 0.2])
df['WeatherIndex'] = 10 + 5 * np.sin(np.linspace(0, 2 * np.pi, months)) + np.random.normal(0, 1, months)

# 4. Feature engineering
def create_full_features(df, lags=12):
    df_feat = pd.DataFrame(index=df.index)
    df_feat['Demand'] = df['Demand']
    for lag in range(1, lags + 1):
        df_feat[f'Lag_{lag}'] = df['Demand'].shift(lag)
    df_feat['ExternalContext'] = df['ExternalContext']
    df_feat['CalendarEffect'] = df['CalendarEffect']
    df_feat['PromotionFlag'] = df['PromotionFlag']
    df_feat['WeatherIndex'] = df['WeatherIndex']
    df_feat.dropna(inplace=True)
    return df_feat

df_features = create_full_features(df)

# 5. Train-test split
train_data = df_features[:-6]
test_data = df_features[-6:]
X_train = train_data.drop(columns='Demand')
y_train = train_data['Demand']
X_test = test_data.drop(columns='Demand')
y_test = test_data['Demand']

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 8. Predict and evaluate
forecast = rf_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, forecast))

# 9. Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Demand'], label='Actual Demand', color='black')
plt.plot(y_test.index, forecast, label='Random Forest Forecast', linestyle='--')
plt.axvline(y_test.index[0], color='gray', linestyle='--')
plt.title(f"Forecast with Full External Contexts (Random Forest)\nRMSE = {rmse:.2f}")
plt.xlabel("Date")
plt.ylabel("Monthly Demand (units)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
