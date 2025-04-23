import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Generate 3-year supply chain dataset with realistic patterns and disruptions
np.random.seed(42)

# Date range: 2019-2022 (weekly data)
dates = pd.date_range(start='2019-01-01', end='2022-01-01', freq='W')

# Create baseline for 5 product categories with different characteristics
categories = ['Electronics', 'Clothing', 'Food', 'Home Goods', 'Health']
products_per_category = 5
total_products = len(categories) * products_per_category

# Product base demand (weekly)
base_demand = np.random.lognormal(mean=5.0, sigma=1.0, size=total_products)

# Lead times (in weeks) - different for each product
base_lead_times = np.random.uniform(1, 8, total_products)

# Cost parameters
unit_costs = np.random.uniform(10, 100, total_products)
holding_cost_rate = 0.25  # 25% annual holding cost
stockout_cost_multiplier = np.random.uniform(1.5, 5, total_products)  # Cost multiplier for stockouts

# Create weekly time index
time_idx = np.arange(len(dates))

# Create product information dataframe
product_info = pd.DataFrame({
    'product_id': range(total_products),
    'category': [cat for cat in categories for _ in range(products_per_category)],
    'base_demand': base_demand,
    'base_lead_time': base_lead_times,
    'unit_cost': unit_costs,
    'stockout_cost_multiplier': stockout_cost_multiplier
})

# Generate time series for each product
demand_data = []
lead_time_data = []
stockout_data = []
inventory_data = []

for product_id in range(total_products):
    category = product_info.loc[product_id, 'category']
    base = product_info.loc[product_id, 'base_demand']
    lead_base = product_info.loc[product_id, 'base_lead_time']
    
    # Create product-specific patterns
    # 1. Seasonality - different by category
    if category == 'Electronics':
        # Holiday peak, summer low
        seasonality = 0.3 * np.sin(2 * np.pi * (time_idx - 45) / 52)
    elif category == 'Clothing':
        # Seasonal changes (spring/fall peaks)
        seasonality = 0.2 * np.sin(4 * np.pi * time_idx / 52) + 0.1 * np.sin(2 * np.pi * time_idx / 52)
    elif category == 'Food':
        # Holiday peak, summer peak
        seasonality = 0.15 * np.sin(2 * np.pi * time_idx / 52) + 0.15 * np.sin(2 * np.pi * (time_idx - 26) / 52)
    elif category == 'Home Goods':
        # Spring peak, fall low
        seasonality = 0.25 * np.sin(2 * np.pi * (time_idx - 13) / 52)
    else:  # Health
        # Winter peak (cold/flu season)
        seasonality = 0.3 * np.cos(2 * np.pi * time_idx / 52)
    
    # 2. Trend - different by product
    trend_factor = np.random.uniform(-0.1, 0.2)  # Some declining, most growing
    trend = trend_factor * time_idx / len(time_idx)
    
    # 3. COVID-19 Effect (starts at week 60, peaks at week 75, gradually diminishes)
    covid_start = 60  # March 2020
    covid_effect = np.zeros_like(time_idx, dtype=float)
    
    covid_mask = time_idx >= covid_start
    covid_time = time_idx[covid_mask] - covid_start
    
    if category == 'Electronics':
        # Increased demand during lockdowns
        covid_effect[covid_mask] = 0.5 * np.exp(-0.05 * (covid_time - 15)**2)
    elif category == 'Clothing':
        # Decreased then recovered
        covid_effect[covid_mask] = -0.4 * np.exp(-0.05 * covid_time) + 0.2 * np.exp(-0.05 * (covid_time - 40)**2)
    elif category == 'Food':
        # Panic buying then new normal
        covid_effect[covid_mask] = 0.8 * np.exp(-0.1 * covid_time) + 0.2
    elif category == 'Home Goods':
        # Delayed peak (home improvement during lockdowns)
        covid_effect[covid_mask] = 0.6 * np.exp(-0.03 * (covid_time - 25)**2)
    else:  # Health
        # Sustained increase
        covid_effect[covid_mask] = 0.7 * (1 - np.exp(-0.1 * covid_time))
    
    # 4. Supply chain disruption effect on lead times (increases after COVID starts)
    lead_disruption = np.zeros_like(time_idx, dtype=float)
    lead_disruption[covid_mask] = 0.5 * (1 - np.exp(-0.05 * covid_time))
    
    # Combine all effects for demand
    demand = base * (1 + seasonality + trend + covid_effect)
    
    # Add noise to demand
    noise_level = 0.15  # 15% noise
    noise = np.random.normal(0, noise_level * base, len(time_idx))
    demand = np.maximum(0, demand + noise)
    
    # Generate lead times with disruption effect
    lead_time = lead_base * (1 + lead_disruption)
    
    # Add noise to lead times
    lead_noise = np.random.normal(0, 0.1 * lead_base, len(time_idx))
    lead_time = np.maximum(1, lead_time + lead_noise)
    
    # Add to dataset
    for week, date in enumerate(dates):
        demand_data.append({
            'date': date,
            'product_id': product_id,
            'category': category,
            'demand': demand[week],
            'lead_time': lead_time[week]
        })

# Create demand and lead time dataframes
supply_chain_df = pd.DataFrame(demand_data)
# supply_chain_df.to_csv('supply_chain_data.csv', index=False)

# Display dataset information
print(f"Supply Chain Dataset: {len(supply_chain_df)} observations")
print(f"Date Range: {supply_chain_df['date'].min()} to {supply_chain_df['date'].max()}")
print(f"Products: {supply_chain_df['product_id'].nunique()}")
print(f"Categories: {supply_chain_df['category'].nunique()}")

# Split dataset
train_cutoff = pd.Timestamp('2021-07-01')
test_cutoff = pd.Timestamp('2022-01-01')

train_df = supply_chain_df[supply_chain_df['date'] < train_cutoff].copy()
test_df = supply_chain_df[(supply_chain_df['date'] >= train_cutoff) & 
                          (supply_chain_df['date'] < test_cutoff)].copy()

print(f"\nTraining set: {len(train_df)} observations, {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Test set: {len(test_df)} observations, {test_df['date'].min()} to {test_df['date'].max()}")

# # Plot aggregated demand by category
# plt.figure(figsize=(15, 8))
# for category in categories:
#     cat_data = supply_chain_df[supply_chain_df['category'] == category]
#     cat_data = cat_data.groupby('date')['demand'].sum().reset_index()
#     plt.plot(cat_data['date'], cat_data['demand'], label=category)

# plt.axvline(x=pd.Timestamp('2020-03-01'), color='r', linestyle='--', alpha=0.7, label='COVID-19 Start')
# plt.axvline(x=train_cutoff, color='g', linestyle='--', alpha=0.7, label='Train/Test Split')
# plt.title('Aggregated Demand by Category', fontsize=14)
# plt.xlabel('Date')
# plt.ylabel('Weekly Demand')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plot lead time trends
# plt.figure(figsize=(15, 6))
# for category in categories:
#     cat_data = supply_chain_df[supply_chain_df['category'] == category]
#     cat_data = cat_data.groupby('date')['lead_time'].mean().reset_index()
#     plt.plot(cat_data['date'], cat_data['lead_time'], label=category)

# plt.axvline(x=pd.Timestamp('2020-03-01'), color='r', linestyle='--', alpha=0.7, label='COVID-19 Start')
# plt.axvline(x=train_cutoff, color='g', linestyle='--', alpha=0.7, label='Train/Test Split')
# plt.title('Average Lead Times by Category', fontsize=14)
# plt.xlabel('Date')
# plt.ylabel('Lead Time (Weeks)')
# plt.legend()
# plt.tight_layout()
# plt.show()

class TraditionalInventoryOptimizer:
    def __init__(self, service_level=0.95):
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)
        self.models = {}
        
    def calculate_eoq(self, demand, unit_cost, holding_cost_rate, order_cost=50):
        """Calculate Economic Order Quantity"""
        annual_demand = demand * 52  # Convert weekly to annual
        annual_holding_cost = unit_cost * holding_cost_rate
        return np.sqrt((2 * annual_demand * order_cost) / annual_holding_cost)
    
    def calculate_safety_stock(self, demand_std, lead_time_avg, service_level=None):
        """Calculate safety stock level"""
        if service_level is None:
            service_level = self.service_level
            
        z_score = stats.norm.ppf(service_level)
        return z_score * demand_std * np.sqrt(lead_time_avg)
    
    def calculate_reorder_point(self, demand_avg, lead_time_avg, safety_stock):
        """Calculate reorder point"""
        return demand_avg * lead_time_avg + safety_stock
    
    def fit(self, train_df, product_info):
        """Fit traditional inventory model"""
        self.product_info = product_info
        
        # Calculate parameters for each product
        for product_id in range(len(product_info)):
            product_data = train_df[train_df['product_id'] == product_id]
            
            # Calculate demand statistics
            demand_avg = product_data['demand'].mean()
            demand_std = product_data['demand'].std()
            
            # Calculate lead time statistics
            lead_time_avg = product_data['lead_time'].mean()
            lead_time_std = product_data['lead_time'].std()
            
            # Get product cost information
            unit_cost = product_info.loc[product_id, 'unit_cost']
            
            # Calculate inventory parameters
            eoq = self.calculate_eoq(demand_avg, unit_cost, holding_cost_rate)
            safety_stock = self.calculate_safety_stock(demand_std, lead_time_avg)
            reorder_point = self.calculate_reorder_point(demand_avg, lead_time_avg, safety_stock)
            
            # Store model parameters
            self.models[product_id] = {
                'demand_avg': demand_avg,
                'demand_std': demand_std,
                'lead_time_avg': lead_time_avg,
                'lead_time_std': lead_time_std,
                'eoq': eoq,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point
            }
            
        return self
    
    def optimize(self, test_df):
        """Generate inventory plans for test period"""
        results = []
        
        for product_id in range(len(self.product_info)):
            # Get model parameters
            params = self.models[product_id]
            product_test = test_df[test_df['product_id'] == product_id]
            
            # Extract parameters
            demand_avg = params['demand_avg']
            eoq = params['eoq']
            safety_stock = params['safety_stock']
            reorder_point = params['reorder_point']
            
            # Generate inventory plan
            for _, row in product_test.iterrows():
                # Calculate required inventory
                expected_demand = demand_avg  # Static forecast
                required_inventory = expected_demand * row['lead_time'] + safety_stock
                order_quantity = eoq if required_inventory > 0 else 0
                
                results.append({
                    'date': row['date'],
                    'product_id': product_id,
                    'category': row['category'],
                    'forecast_demand': expected_demand,
                    'actual_demand': row['demand'],
                    'lead_time': row['lead_time'],
                    'safety_stock': safety_stock,
                    'reorder_point': reorder_point,
                    'order_quantity': order_quantity,
                    'required_inventory': required_inventory
                })
                
        return pd.DataFrame(results)



class ContextAwareInventoryOptimizer:
    def __init__(self, service_level_base=0.95):
        self.service_level_base = service_level_base
        self.models = {}
        self.forecast_models = {}
        self.forecast_errors = {}
        self.category_buffer = {
            'Electronics': 0.15,
            'Clothing': 0.10,
            'Food': 0.20,
            'Home Goods': 0.10,
            'Health': 0.05
        }

    def time_weighted_average(self, series, decay=0.02):
        weights = np.exp(-decay * np.arange(len(series))[::-1])
        return np.sum(series * weights) / np.sum(weights)

    def compute_trend(self, series):
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope

    def detect_drift(self, base_series, current_series):
        base_mean = np.mean(base_series)
        current_mean = np.mean(current_series)
        return np.abs(current_mean - base_mean) / base_mean

    def fabricate_external_signals(self, df):
        df['promo_flag'] = np.random.binomial(1, 0.1, size=len(df))
        df['holiday_flag'] = df['date'].dt.month.isin([11, 12]).astype(int)
        df['weather_temp'] = 15 + 10 * np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
        df['economic_index'] = 100 + 5 * np.random.randn(len(df))
        return df

    def train_forecast_model(self, product_df):
        df = self.fabricate_external_signals(product_df.copy())
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['lag_demand'] = df['demand'].shift(1)
        df = df.dropna()

        features = ['weekofyear', 'lead_time', 'lag_demand', 'promo_flag', 'holiday_flag', 'weather_temp', 'economic_index']
        X = df[features]
        y = df['demand']
        model = GradientBoostingRegressor().fit(X, y)
        return model, features

    def fit(self, train_df, product_info):
        self.product_info = product_info

        for product_id in range(len(product_info)):
            product_data = train_df[train_df['product_id'] == product_id].copy()
            product_data['weekofyear'] = product_data['date'].dt.isocalendar().week
            product_data = self.fabricate_external_signals(product_data)

            demand_series = product_data['demand'].values
            lead_time_series = product_data['lead_time'].values

            contextual_avg = self.time_weighted_average(demand_series)
            demand_std = np.std(demand_series)
            lead_time_avg = np.mean(lead_time_series)

            volatility = np.std(demand_series[-8:])
            volatility_factor = np.clip(volatility / contextual_avg, 0.5, 2.0)

            adaptive_safety_stock = 1.65 * demand_std * np.sqrt(lead_time_avg) * volatility_factor

            recent_demand = product_data['demand'].tail(12).values
            trend = self.compute_trend(recent_demand)

            reorder_point = contextual_avg * lead_time_avg + adaptive_safety_stock + trend * lead_time_avg

            self.models[product_id] = {
                'demand_avg': contextual_avg,
                'demand_std': demand_std,
                'lead_time_avg': lead_time_avg,
                'volatility': volatility,
                'adaptive_safety_stock': adaptive_safety_stock,
                'reorder_point': reorder_point,
                'trend': trend,
                'max_demand': np.percentile(demand_series, 95),
                'lead_time_cap': np.max(lead_time_series)
            }

            model, features = self.train_forecast_model(product_data)
            self.forecast_models[product_id] = {'model': model, 'features': features}

    def optimize(self, test_df, retrain_frequency=4):
        results = []
        inventory_tracker = {}

        for product_id in range(len(self.product_info)):
            product_test = test_df[test_df['product_id'] == product_id].copy()
            product_test['weekofyear'] = product_test['date'].dt.isocalendar().week
            product_test = self.fabricate_external_signals(product_test)

            params = self.models[product_id]
            model_bundle = self.forecast_models[product_id]
            model = model_bundle['model']
            features = model_bundle['features']

            prev_demand = params['demand_avg']
            forecasts = []
            current_inventory = 0

            for i, row in product_test.iterrows():
                if i % retrain_frequency == 0 and i >= retrain_frequency:
                    retrain_data = product_test.iloc[max(0, i - 12):i].copy()
                    retrain_data['lag_demand'] = retrain_data['demand'].shift(1)
                    retrain_data = retrain_data.dropna()

                    if not retrain_data.empty:  # Check if retrain_data is not empty
                        X_retrain = retrain_data[features]
                        y_retrain = retrain_data['demand']
                        model.fit(X_retrain, y_retrain)

                row_input = row.copy()
                row_input['lag_demand'] = prev_demand
                X = pd.DataFrame([row_input[features].values], columns=features)
                expected_demand = model.predict(X)[0]
                prev_demand = row['demand']

                expected_demand = max(expected_demand, 0.85 * params['demand_avg'])
                buffer_factor = self.category_buffer.get(row['category'], 0.1)
                expected_demand *= (1 + buffer_factor)

                urgency_weight = np.log1p(self.product_info.loc[product_id, 'stockout_cost_multiplier']) / 2
                urgency_factor = np.clip((row['lead_time'] / params['lead_time_avg']) - 1, 0.9, 2.0)

                decay_factor = np.exp(-i / len(product_test))
                adjusted_safety_stock = params['adaptive_safety_stock'] * decay_factor

                lookahead_weeks = row['lead_time'] + 1
                forecast_horizon = expected_demand * lookahead_weeks
                required_inventory = forecast_horizon + adjusted_safety_stock

                inventory_target = expected_demand * 13
                max_inventory = inventory_target * row['lead_time'] / 52
                required_inventory = min(required_inventory, max_inventory)

                max_order = params['max_demand'] * params['lead_time_cap']
                min_order = 0.75 * params['demand_avg'] * row['lead_time']

                # Inventory-aware decision logic
                gap = required_inventory - current_inventory
                inventory_penalty = np.clip(current_inventory / max_inventory, 0.5, 1.0)
                order_quantity = max(min_order, min(gap * urgency_factor * urgency_weight * inventory_penalty, max_order))

                # Suppress orders if inventory is healthy
                if current_inventory > forecast_horizon + adjusted_safety_stock:
                    order_quantity = 0

                # Simulate inventory update
                current_inventory += order_quantity - row['demand']
                current_inventory = max(current_inventory, 0)

                forecasts.append((expected_demand, row['demand']))

                results.append({
                    'date': row['date'],
                    'product_id': product_id,
                    'category': row['category'],
                    'forecast_demand': expected_demand,
                    'actual_demand': row['demand'],
                    'lead_time': row['lead_time'],
                    'safety_stock': adjusted_safety_stock,
                    'reorder_point': params['reorder_point'],
                    'order_quantity': order_quantity,
                    'required_inventory': required_inventory,
                    'current_inventory': current_inventory
                })

            forecasts = np.array(forecasts)
            if len(forecasts) > 0:
                forecast_values, actual_values = forecasts[:, 0], forecasts[:, 1]
                mape = mean_absolute_percentage_error(actual_values, forecast_values)
                self.forecast_errors[product_id] = mape

        return pd.DataFrame(results)

# Apply traditional inventory optimization
traditional_optimizer = TraditionalInventoryOptimizer(service_level=0.95)
traditional_optimizer.fit(train_df, product_info)

# Pass traditional models to the context-aware optimizer
context_optimizer = ContextAwareInventoryOptimizer(
    service_level_base=0.95
)
context_optimizer.fit(train_df, product_info)
context_aware_results = context_optimizer.optimize(test_df)

def simulate_inventory_operations(plan_df, product_info, initial_inventory=None):
    """Simulate inventory operations based on the inventory plan"""
    # Initialize results
    simulation_results = []
    
    # Initialize inventory levels
    if initial_inventory is None:
        # Default to safety stock levels
        inventory = {}
        for product_id in plan_df['product_id'].unique():
            product_plan = plan_df[plan_df['product_id'] == product_id].iloc[0]
            inventory[product_id] = product_plan['safety_stock']
    else:
        inventory = initial_inventory.copy()
    
    # Simulate each time period
    for date, date_group in plan_df.groupby('date'):
        for _, row in date_group.iterrows():
            product_id = row['product_id']
            
            # Get current inventory level
            current_inventory = inventory.get(product_id, 0)
            
            # Process demand
            actual_demand = row['actual_demand']
            fulfilled_demand = min(current_inventory, actual_demand)
            stockout = max(0, actual_demand - fulfilled_demand)
            ending_inventory = max(0, current_inventory - actual_demand)
            
            # Process any orders that arrive today
            # (in a real simulation we would track orders in transit)
            # For simplicity, assume orders arrive instantly with lead time considered in planning
            order_quantity = row['order_quantity']
            
            # Update inventory
            new_inventory = ending_inventory + order_quantity
            inventory[product_id] = new_inventory
            
            # Calculate costs
            unit_cost = product_info.loc[product_id, 'unit_cost']
            holding_cost = ending_inventory * (unit_cost * holding_cost_rate / 52)  # Weekly holding cost
            stockout_cost = stockout * unit_cost * product_info.loc[product_id, 'stockout_cost_multiplier']
            order_cost = 50 if order_quantity > 0 else 0  # Fixed ordering cost
            total_cost = holding_cost + stockout_cost + order_cost
            
            # Calculate service level
            service_level = fulfilled_demand / actual_demand if actual_demand > 0 else 1.0
            
            # Store results
            simulation_results.append({
                'date': date,
                'product_id': product_id,
                'category': row['category'],
                'starting_inventory': current_inventory,
                'actual_demand': actual_demand,
                'fulfilled_demand': fulfilled_demand,
                'stockout': stockout,
                'ending_inventory': ending_inventory,
                'order_quantity': order_quantity,
                'new_inventory': new_inventory,
                'holding_cost': holding_cost,
                'stockout_cost': stockout_cost,
                'order_cost': order_cost,
                'total_cost': total_cost,
                'service_level': service_level
            })
    
    return pd.DataFrame(simulation_results)

# Simulate inventory operations for both approaches
traditional_simulation = simulate_inventory_operations(traditional_optimizer.optimize(test_df), product_info)
context_aware_simulation = simulate_inventory_operations(context_aware_results, product_info)

# Calculate performance metrics
def calculate_performance_metrics(simulation_df):
    """Calculate key performance metrics from simulation results"""
    # Aggregate metrics
    total_holding_cost = simulation_df['holding_cost'].sum()
    total_stockout_cost = simulation_df['stockout_cost'].sum()
    total_order_cost = simulation_df['order_cost'].sum()
    total_cost = simulation_df['total_cost'].sum()
    
    # Calculate service level (fill rate)
    total_demand = simulation_df['actual_demand'].sum()
    fulfilled_demand = simulation_df['fulfilled_demand'].sum()
    overall_service_level = fulfilled_demand / total_demand if total_demand > 0 else 1.0
    
    # Calculate stockout frequency
    stockout_events = (simulation_df['stockout'] > 0).sum()
    total_events = len(simulation_df)
    stockout_frequency = stockout_events / total_events
    
    # Calculate average inventory
    avg_inventory = simulation_df['ending_inventory'].mean()
    
    # Calculate inventory turns
    total_cogs = simulation_df['fulfilled_demand'].sum() * simulation_df.groupby('product_id')['actual_demand'].mean().mean()
    inventory_turns = total_cogs / avg_inventory if avg_inventory > 0 else 0
    
    # Calculate metrics by category
    category_metrics = {}
    for category in simulation_df['category'].unique():
        cat_data = simulation_df[simulation_df['category'] == category]
        
        cat_service_level = cat_data['fulfilled_demand'].sum() / cat_data['actual_demand'].sum() if cat_data['actual_demand'].sum() > 0 else 1.0
        cat_stockout_freq = (cat_data['stockout'] > 0).sum() / len(cat_data)
        cat_avg_inventory = cat_data['ending_inventory'].mean()
        cat_total_cost = cat_data['total_cost'].sum()
        
        category_metrics[category] = {
            'service_level': cat_service_level,
            'stockout_frequency': cat_stockout_freq,
            'avg_inventory': cat_avg_inventory,
            'total_cost': cat_total_cost
        }
    
    return {
        'total_holding_cost': total_holding_cost,
        'total_stockout_cost': total_stockout_cost,
        'total_order_cost': total_order_cost,
        'total_cost': total_cost,
        'service_level': overall_service_level,
        'stockout_frequency': stockout_frequency,
        'avg_inventory': avg_inventory,
        'inventory_turns': inventory_turns,
        'category_metrics': category_metrics
    }

# Calculate metrics for both approaches
traditional_metrics = calculate_performance_metrics(traditional_simulation)
context_aware_metrics = calculate_performance_metrics(context_aware_simulation)

# Display overall results
results_comparison = pd.DataFrame({
    'Metric': ['Total Cost', 'Service Level', 'Stockout Frequency', 'Avg Inventory', 'Inventory Turns'],
    'Traditional': [
        traditional_metrics['total_cost'],
        traditional_metrics['service_level'],
        traditional_metrics['stockout_frequency'],
        traditional_metrics['avg_inventory'],
        traditional_metrics['inventory_turns']
    ],
    'Context-Aware': [
        context_aware_metrics['total_cost'],
        context_aware_metrics['service_level'],
        context_aware_metrics['stockout_frequency'],
        context_aware_metrics['avg_inventory'],
        context_aware_metrics['inventory_turns']
    ]
})

results_comparison['Improvement'] = [
    f"{(1 - context_aware_metrics['total_cost'] / traditional_metrics['total_cost']) * 100:.2f}%",
    f"{(context_aware_metrics['service_level'] - traditional_metrics['service_level']) * 100:.2f}%",
    f"{(1 - context_aware_metrics['stockout_frequency'] / traditional_metrics['stockout_frequency']) * 100:.2f}%",
    f"{(1 - context_aware_metrics['avg_inventory'] / traditional_metrics['avg_inventory']) * 100:.2f}%",
    f"{(context_aware_metrics['inventory_turns'] / traditional_metrics['inventory_turns'] - 1) * 100:.2f}%"
]

print("Supply Chain Performance Comparison:")
print(results_comparison)

# Display category-specific results
category_comparison = []
for category in product_info['category'].unique():
    trad_cat = traditional_metrics['category_metrics'][category]
    context_cat = context_aware_metrics['category_metrics'][category]
    
    category_comparison.append({
        'Category': category,
        'Traditional Service Level': trad_cat['service_level'],
        'Context-Aware Service Level': context_cat['service_level'],
        'Service Level Improvement': (context_cat['service_level'] - trad_cat['service_level']) * 100,
        'Traditional Cost': trad_cat['total_cost'],
        'Context-Aware Cost': context_cat['total_cost'],
        'Cost Reduction': (1 - context_cat['total_cost'] / trad_cat['total_cost']) * 100
    })
    
category_df = pd.DataFrame(category_comparison)
print("\nCategory-Specific Improvements:")
print(category_df)

def compare_with_traditional(context_df, traditional_df):
    """
    Compare context-aware vs traditional inventory strategies.

    Traditional method explanation:
    - Uses fixed reorder point and lead time demand estimates.
    - Assumes static safety stock with no trend/volatility adjustment.
    - No external signals or adaptive buffer logic.
    """
    
    def aggregate_metrics(df):
        return df.groupby('product_id').agg(
            total_cost=('order_quantity', 'sum'),
            avg_inventory=('starting_inventory', 'mean'),
            stockouts=('stockout', lambda x: (x == 0).sum()),
            service_level=('service_level', lambda x: (x > 0).mean())
        ).reset_index()

    context_metrics = aggregate_metrics(context_df)
    traditional_metrics = aggregate_metrics(traditional_df)

    comparison = context_metrics.merge(traditional_metrics, on='product_id', suffixes=('_context', '_traditional'))
    comparison['cost_saving'] = comparison['total_cost_traditional'] - comparison['total_cost_context']
    comparison['inventory_reduction'] = comparison['avg_inventory_traditional'] - comparison['avg_inventory_context']
    comparison['stockout_diff'] = comparison['stockouts_traditional'] - comparison['stockouts_context']
    comparison['service_diff'] = comparison['service_level_context'] - comparison['service_level_traditional']

    return comparison


def evaluate_forecast_accuracy(forecast_df, label='context'):
    """
    Calculate forecast accuracy metrics from a forecast DataFrame.
    """
    y_true = forecast_df['actual_demand']
    y_pred = forecast_df['forecast_demand']

    return {
        f'{label}_MAE': mean_absolute_error(y_true, y_pred),
        f'{label}_RMSE': root_mean_squared_error(y_true, y_pred),
        f'{label}_MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }

def plot_comparison_summary(comparison_df):
    """
    Generate bar plots comparing context-aware vs traditional performance metrics.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    comparison_df.plot(x='product_id', y=['total_cost_context', 'total_cost_traditional'], kind='bar', ax=axs[0, 0], title='Total Cost')
    comparison_df.plot(x='product_id', y=['avg_inventory_context', 'avg_inventory_traditional'], kind='bar', ax=axs[0, 1], title='Avg Inventory')
    comparison_df.plot(x='product_id', y=['stockouts_context', 'stockouts_traditional'], kind='bar', ax=axs[1, 0], title='Stockouts')
    comparison_df.plot(x='product_id', y=['service_level_context', 'service_level_traditional'], kind='bar', ax=axs[1, 1], title='Service Level')

    for ax in axs.flat:
        ax.set_xlabel('Product ID')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def generate_forecast_set(optimizer, test_df):
    forecasts = []
    for product_id in range(len(optimizer.product_info)):
        product_test = test_df[test_df['product_id'] == product_id].copy()
        product_test['weekofyear'] = product_test['date'].dt.isocalendar().week
        product_test = optimizer.fabricate_external_signals(product_test)

        params = optimizer.models[product_id]
        model_bundle = optimizer.forecast_models[product_id]
        model = model_bundle['model']
        features = model_bundle['features']

        prev_demand = params['demand_avg']

        for _, row in product_test.iterrows():
            row_input = row.copy()
            row_input['lag_demand'] = prev_demand
            X = pd.DataFrame([row_input[features].values], columns=features)
            expected_demand = model.predict(X)[0]
            prev_demand = row['demand']

            forecasts.append({
                'date': row['date'],
                'product_id': row['product_id'],
                'category': row['category'],
                'forecast_demand': expected_demand,
                'actual_demand': row['demand']
            })
    return pd.DataFrame(forecasts)

def generate_traditional_forecast(test_df):
    """
    Generate simple traditional forecasts using moving average.
    """
    forecast_rows = []
    for product_id in test_df['product_id'].unique():
        product_df = test_df[test_df['product_id'] == product_id].copy()
        product_df = product_df.sort_values('date')
        product_df['forecast_demand'] = product_df['demand'].rolling(window=4, min_periods=1).mean().shift(1)
        product_df['forecast_demand'].fillna(method='bfill', inplace=True)
        forecast_rows.append(product_df[['date', 'product_id', 'category', 'forecast_demand', 'demand']])

    forecast_df = pd.concat(forecast_rows)
    forecast_df.rename(columns={'demand': 'actual_demand'}, inplace=True)
    return forecast_df.reset_index(drop=True)

def evaluate_forecast_comparison(context_forecast_df, traditional_forecast_df):
    """
    Compare forecast accuracy between context-aware and traditional forecasts.
    """
    context_metrics = evaluate_forecast_accuracy(context_forecast_df, label='context')
    traditional_metrics = evaluate_forecast_accuracy(traditional_forecast_df, label='traditional')

    return {**context_metrics, **traditional_metrics}

def plot_forecast_vs_actual_by_category(forecast_df, title_prefix='Forecast Evaluation'):
    categories = forecast_df['category'].unique()
    fig, axs = plt.subplots(len(categories), 1, figsize=(15, 4 * len(categories)), sharex=True)

    if len(categories) == 1:
        axs = [axs]

    for i, category in enumerate(categories):
        cat_df = forecast_df[forecast_df['category'] == category]
        cat_df = cat_df.groupby('date')[['actual_demand', 'forecast_demand']].sum()

        axs[i].plot(cat_df.index, cat_df['actual_demand'], label='Actual Demand', color='black', linewidth=2)
        axs[i].plot(cat_df.index, cat_df['forecast_demand'], label='Forecast Demand', linestyle='--', color='tab:blue')
        axs[i].set_title(f"{title_prefix} for {category}")
        axs[i].set_ylabel("Weekly Demand")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()


# 3. Compare performance
comparison = compare_with_traditional(context_aware_simulation, traditional_simulation)
# plot_comparison_summary(comparison)

# context_optimizer.optimize(test_df)

# Generate forecast sets for context-aware and traditional methods
# plot_forecast_vs_actual_by_category(forecast_df)
context_forecast_df = generate_forecast_set(context_optimizer, test_df)
traditional_forecast_df = generate_traditional_forecast(test_df)

forecast_comparison = evaluate_forecast_comparison(context_forecast_df, traditional_forecast_df)
print(forecast_comparison)



# 4. Evaluate forecast accuracy
# context_accuracy = evaluate_forecast_accuracy(context_aware_simulation, label='context')
# traditional_accuracy = evaluate_forecast_accuracy(traditional_simulation, label='traditional')


# print("\nForecast Accuracy Results:")
# print("Context-Aware Accuracy:", context_accuracy)
# print("Traditional Accuracy:", traditional_accuracy)

# Plot key metrics comparison
plt.figure(figsize=(15, 10))

# Plot 1: Service Level by Category
plt.subplot(221)
x = range(len(category_df))
width = 0.35
plt.bar([i - width/2 for i in x], category_df['Traditional Service Level'], width, label='Traditional')
plt.bar([i + width/2 for i in x], category_df['Context-Aware Service Level'], width, label='Context-Aware')
plt.xlabel('Category')
plt.ylabel('Service Level')
plt.title('Service Level by Category')
plt.xticks(x, category_df['Category'])
plt.legend()

# Plot 2: Total Cost by Category
plt.subplot(222)
plt.bar([i - width/2 for i in x], category_df['Traditional Cost'], width, label='Traditional')
plt.bar([i + width/2 for i in x], category_df['Context-Aware Cost'], width, label='Context-Aware')
plt.xlabel('Category')
plt.ylabel('Total Cost')
plt.title('Total Cost by Category')
plt.xticks(x, category_df['Category'])
plt.legend()

# Plot 3: Stockouts Over Time
plt.subplot(223)
stockouts_by_week = traditional_simulation.groupby('date')['stockout'].sum()
context_stockouts_by_week = context_aware_simulation.groupby('date')['stockout'].sum()
plt.plot(stockouts_by_week.index, stockouts_by_week, label='Traditional')
plt.plot(context_stockouts_by_week.index, context_stockouts_by_week, label='Context-Aware')
plt.xlabel('Date')
plt.ylabel('Total Stockouts')
plt.title('Stockouts Over Time')
plt.legend()

# Plot 4: Inventory Levels Over Time
plt.subplot(224)
inventory_by_week = traditional_simulation.groupby('date')['ending_inventory'].mean()
context_inventory_by_week = context_aware_simulation.groupby('date')['ending_inventory'].mean()
plt.plot(inventory_by_week.index, inventory_by_week, label='Traditional')
plt.plot(context_inventory_by_week.index, context_inventory_by_week, label='Context-Aware')
plt.xlabel('Date')
plt.ylabel('Average Inventory Level')
plt.title('Inventory Levels Over Time')
plt.legend()

plt.tight_layout()
plt.show()