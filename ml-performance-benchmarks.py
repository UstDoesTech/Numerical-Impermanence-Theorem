import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

#===========================================================
# 1. Feature Scaling Benchmark
#===========================================================

def feature_scaling_benchmark():
    """Benchmark different feature scaling approaches with and without
    numerical impermanence consideration."""
    
    print("Running Feature Scaling Benchmark...")
    
    # Generate synthetic data with mixed scales and a domain shift
    def generate_mixed_scale_data(n_samples=1000, n_features=10, domain='source'):
        # Feature magnitudes across different scales
        magnitudes = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 
                              0.05, 0.5, 5.0, 50.0])
        
        # Generate base features
        X = np.random.randn(n_samples, n_features)
        
        # Scale features by their magnitudes
        X = X * magnitudes
        
        # True coefficients are inversely proportional to magnitude
        # so each feature contributes equally to the target
        true_coef = 1.0 / magnitudes
        
        # Target with some noise
        y = np.dot(X, true_coef) + np.random.normal(0, 0.1, n_samples)
        
        # For target domain, shift the distribution of some features
        if domain == 'target':
            # Shift the first 5 features
            X[:, :5] = X[:, :5] + np.random.normal(5, 1, (n_samples, 5))
            
            # Change the scale of features 3-7
            X[:, 3:8] = X[:, 3:8] * 2.0
        
        return X, y, true_coef

    # Generate source and target domain data
    X_source, y_source, true_coef = generate_mixed_scale_data(domain='source')
    X_target, y_target, _ = generate_mixed_scale_data(domain='target')

    # Split source data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size=0.2)

    # Context-unaware approach: Global standardization
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    X_target_std = std_scaler.transform(X_target)

    # Context-unaware approach: Min-max scaling
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)
    X_target_minmax = minmax_scaler.transform(X_target)

    # Context-aware approach: Feature-specific normalization
    # This normalizes each feature differently based on its magnitude context
    class ContextAwareScaler:
        def __init__(self):
            self.feature_contexts = None
            self.scalers = {}
            
        def fit_transform(self, X):
            n_samples, n_features = X.shape
            self.feature_contexts = np.zeros(n_features)
            
            # Determine the context (magnitude) of each feature
            for i in range(n_features):
                magnitude = np.log10(np.abs(X[:, i]).mean() + 1e-10)
                self.feature_contexts[i] = magnitude
                
                # Group features by magnitude context
                context_group = int(np.round(magnitude))
                
                if context_group not in self.scalers:
                    self.scalers[context_group] = StandardScaler()
                    
            # Transform each feature using appropriate scaler
            X_transformed = X.copy()
            for i in range(n_features):
                context_group = int(np.round(self.feature_contexts[i]))
                X_transformed[:, i] = self.scalers[context_group].fit_transform(X[:, i].reshape(-1, 1)).flatten()
                
            return X_transformed
        
        def transform(self, X):
            n_samples, n_features = X.shape
            X_transformed = X.copy()
            
            for i in range(n_features):
                context_group = int(np.round(self.feature_contexts[i]))
                X_transformed[:, i] = self.scalers[context_group].transform(X[:, i].reshape(-1, 1)).flatten()
                
            return X_transformed

    # Apply context-aware scaling
    ca_scaler = ContextAwareScaler()
    X_train_ca = ca_scaler.fit_transform(X_train)
    X_test_ca = ca_scaler.transform(X_test)
    X_target_ca = ca_scaler.transform(X_target)

    # Evaluate accuracy of coefficients under each normalization method
    def evaluate_coefficient_recovery(X_train_scaled, y_train, true_coef, scaler_name):
        # Fit a linear model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # For fair comparison, we need to transform coefficients back to original scale
        if scaler_name == "Standard":
            # For standard scaler: coef * std
            recovered_coef = model.coef_ * std_scaler.scale_
        elif scaler_name == "MinMax":
            # For min-max scaler: coef * (max - min)
            recovered_coef = model.coef_ * (minmax_scaler.data_max_ - minmax_scaler.data_min_)
        else:
            # Context-aware scaling is more complex to invert, we'll use raw comparison
            recovered_coef = model.coef_
        
        # Calculate coefficient error
        coef_error = np.mean(np.abs((recovered_coef - true_coef) / true_coef))
        
        return model, coef_error

    # Evaluate models on test and target data
    def evaluate_performance(model, X_test_scaled, y_test, X_target_scaled, y_target):
        # Test domain performance
        y_pred_test = model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        # Target domain performance
        y_pred_target = model.predict(X_target_scaled)
        target_mse = mean_squared_error(y_target, y_pred_target)
        
        return test_mse, target_mse

    # Evaluate all methods
    results = []

    # Standard scaling
    model_std, coef_error_std = evaluate_coefficient_recovery(X_train_std, y_train, true_coef, "Standard")
    test_mse_std, target_mse_std = evaluate_performance(model_std, X_test_std, y_test, X_target_std, y_target)
    results.append(["Standard Scaling", coef_error_std, test_mse_std, target_mse_std])

    # Min-max scaling
    model_minmax, coef_error_minmax = evaluate_coefficient_recovery(X_train_minmax, y_train, true_coef, "MinMax")
    test_mse_minmax, target_mse_minmax = evaluate_performance(model_minmax, X_test_minmax, y_test, X_target_minmax, y_target)
    results.append(["Min-Max Scaling", coef_error_minmax, test_mse_minmax, target_mse_minmax])

    # Context-aware scaling
    model_ca, coef_error_ca = evaluate_coefficient_recovery(X_train_ca, y_train, true_coef, "ContextAware")
    test_mse_ca, target_mse_ca = evaluate_performance(model_ca, X_test_ca, y_test, X_target_ca, y_target)
    results.append(["Context-Aware Scaling", coef_error_ca, test_mse_ca, target_mse_ca])

    # Format results
    results_df = pd.DataFrame(results, columns=["Method", "Coefficient Error", "Test MSE", "Target MSE"])
    print("\nFeature Scaling Benchmark Results:")
    print(results_df)

    # Calculate improvement percentages
    std_to_ca_test_improvement = (test_mse_std - test_mse_ca) / test_mse_std * 100
    std_to_ca_target_improvement = (target_mse_std - target_mse_ca) / target_mse_std * 100

    minmax_to_ca_test_improvement = (test_mse_minmax - test_mse_ca) / test_mse_minmax * 100
    minmax_to_ca_target_improvement = (target_mse_minmax - target_mse_ca) / target_mse_minmax * 100

    print(f"\nImprovement from Standard to Context-Aware: {std_to_ca_test_improvement:.2f}% (Test), {std_to_ca_target_improvement:.2f}% (Target)")
    print(f"Improvement from MinMax to Context-Aware: {minmax_to_ca_test_improvement:.2f}% (Test), {minmax_to_ca_target_improvement:.2f}% (Target)")
    
    return {
        "std_mse": test_mse_std,
        "minmax_mse": test_mse_minmax,
        "ca_mse": test_mse_ca,
        "std_target_mse": target_mse_std,
        "minmax_target_mse": target_mse_minmax,
        "ca_target_mse": target_mse_ca,
        "std_to_ca_improvement": std_to_ca_target_improvement,
        "minmax_to_ca_improvement": minmax_to_ca_target_improvement
    }

#===========================================================
# 2. Concept Drift Benchmark
#===========================================================

def concept_drift_benchmark():
    """Benchmark models under concept drift with and without
    numerical impermanence consideration."""
    
    print("\nRunning Concept Drift Benchmark...")
    
    # Generate time series data with concept drift
    def generate_drift_data(n_samples=1000, n_features=5, n_drifts=3):
        X = np.random.randn(n_samples, n_features)
        
        # Generate true coefficients that change over time (concept drift)
        segment_size = n_samples // n_drifts
        true_coefs = np.zeros((n_drifts, n_features))
        
        # Each drift period has different coefficients
        for i in range(n_drifts):
            true_coefs[i] = np.random.uniform(-1, 1, n_features)
        
        # Generate target with drift
        y = np.zeros(n_samples)
        for i in range(n_drifts):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, n_samples)
            segment_X = X[start_idx:end_idx]
            y[start_idx:end_idx] = np.dot(segment_X, true_coefs[i]) + np.random.normal(0, 0.1, end_idx - start_idx)
        
        return X, y, true_coefs, segment_size

    # Generate data with concept drift
    X_drift, y_drift, true_coefs_drift, segment_size = generate_drift_data(n_samples=3000)

    # Context-unaware approach: Global model without adaptation
    def train_global_model(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    # Context-unaware approach: Sliding window model
    def train_sliding_window_model(X, y, window_size):
        model = LinearRegression()
        model.fit(X[-window_size:], y[-window_size:])
        return model

    # Context-aware approach: Drift detection and adaptation
    class ConceptDriftModel:
        def __init__(self, base_model=LinearRegression(), window_size=200, 
                     drift_threshold=0.05, min_instances=50):
            self.base_model = base_model
            self.window_size = window_size
            self.drift_threshold = drift_threshold
            self.min_instances = min_instances
            self.models = []
            self.drift_points = []
            self.error_history = []
            
        def detect_drift(self, X, y, current_model):
            """Simple drift detection based on error increase"""
            y_pred = current_model.predict(X)
            error = mean_squared_error(y, y_pred)
            self.error_history.append(error)
            
            # Need some history to detect drift
            if len(self.error_history) < 2:
                return False
                
            # If error increased significantly, detect drift
            if error > self.error_history[-2] * (1 + self.drift_threshold):
                return True
                
            return False
            
        def fit(self, X, y):
            """Train the model with drift detection"""
            n_samples = X.shape[0]
            current_X = X[:self.min_instances]
            current_y = y[:self.min_instances]
            
            # Train initial model
            current_model = self.base_model.__class__()
            current_model.fit(current_X, current_y)
            self.models.append(current_model)
            
            # Process remaining data with drift detection
            for i in range(self.min_instances, n_samples, self.window_size):
                end_idx = min(i + self.window_size, n_samples)
                batch_X = X[i:end_idx]
                batch_y = y[i:end_idx]
                
                # Check for drift
                if self.detect_drift(batch_X, batch_y, current_model):
                    # Drift detected, train new model
                    new_model = self.base_model.__class__()
                    # Use recent data for the new model
                    recent_start = max(0, i - self.window_size)
                    new_model.fit(X[recent_start:end_idx], y[recent_start:end_idx])
                    
                    self.models.append(new_model)
                    self.drift_points.append(i)
                    current_model = new_model
                
                # Update current model with new data
                # For LinearRegression we need to retrain, for incremental models we could use partial_fit
                window_start = max(0, end_idx - self.window_size*2)
                current_model = self.base_model.__class__()
                current_model.fit(X[window_start:end_idx], y[window_start:end_idx])
                
                # Replace the latest model
                self.models[-1] = current_model
        
        def predict(self, X):
            """Predict using the most recent model"""
            if not self.models:
                raise ValueError("Model must be fitted before predicting")
            
            return self.models[-1].predict(X)

    # Evaluate models on drift data
    def evaluate_drift_models(X, y, segment_size):
        n_samples = X.shape[0]
        n_drifts = n_samples // segment_size
        
        # Prepare for storing results
        all_results = []
        
        # Sliding window sizes to try
        window_sizes = [100, 300, 500]
        
        # Train and evaluate each model
        for model_name, model_func, params in [
            ("Global Model", train_global_model, {}),
            *[("Sliding Window " + str(w), train_sliding_window_model, {"window_size": w}) for w in window_sizes],
            ("Context-Aware Model", ConceptDriftModel, {})
        ]:
            drift_errors = []
            
            for drift_idx in range(n_drifts):
                # Define drift period indices
                start_idx = drift_idx * segment_size
                end_idx = min((drift_idx + 1) * segment_size, n_samples)
                
                if model_name == "Global Model":
                    # Global model uses all available data up to current point
                    train_idx = min(end_idx, n_samples - segment_size)
                    model = model_func(X[:train_idx], y[:train_idx])
                    
                elif model_name.startswith("Sliding Window"):
                    # Sliding window only uses recent data
                    window_size = params["window_size"]
                    train_start = max(0, end_idx - window_size)
                    model = model_func(X[train_start:end_idx], y[train_start:end_idx], window_size)
                    
                elif model_name == "Context-Aware Model":
                    # Context-aware model detects and adapts to drift
                    model = ConceptDriftModel(**params)
                    model.fit(X[:end_idx], y[:end_idx])
                
                # Evaluate on next segment (future data after this drift period)
                test_start = end_idx
                test_end = min(test_start + segment_size // 2, n_samples)
                
                if test_start < test_end:
                    y_pred = model.predict(X[test_start:test_end])
                    mse = mean_squared_error(y[test_start:test_end], y_pred)
                    drift_errors.append(mse)
            
            # Store results for this model
            all_results.append({
                "Model": model_name,
                "Average MSE": np.mean(drift_errors),
                "MSE by Drift": drift_errors
            })
        
        return all_results

    # Run benchmark
    drift_results = evaluate_drift_models(X_drift, y_drift, segment_size)

    # Format drift results
    drift_results_df = pd.DataFrame([
        {"Model": r["Model"], "Average MSE": r["Average MSE"]} 
        for r in drift_results
    ])
    print("\nConcept Drift Benchmark Results:")
    print(drift_results_df)

    # Calculate improvement from best non-CA to CA
    non_ca_models = [r for r in drift_results if r["Model"] != "Context-Aware Model"]
    best_non_ca = min(non_ca_models, key=lambda x: x["Average MSE"])
    ca_model = next(r for r in drift_results if r["Model"] == "Context-Aware Model")

    drift_improvement = (best_non_ca["Average MSE"] - ca_model["Average MSE"]) / best_non_ca["Average MSE"] * 100
    print(f"\nImprovement from {best_non_ca['Model']} to Context-Aware Model: {drift_improvement:.2f}%")
    
    return {
        "drift_results": drift_results,
        "best_non_ca_model": best_non_ca["Model"],
        "best_non_ca_mse": best_non_ca["Average MSE"],
        "ca_model_mse": ca_model["Average MSE"],
        "drift_improvement": drift_improvement
    }

#===========================================================
# 3. Transfer Learning Benchmark
#===========================================================

def transfer_learning_benchmark():
    """Benchmark transfer learning with and without numerical impermanence consideration."""
    
    print("\nRunning Transfer Learning Benchmark...")
    
    # Generate synthetic data for transfer learning
    def generate_transfer_data(n_samples=1000, n_features=20, difficulty=0.5):
        """Generate source and target domains with controlled similarity"""
        # Source domain data
        X_source = np.random.randn(n_samples, n_features)
        
        # Define true model with interaction terms
        core_features = 5
        core_coef = np.random.uniform(-1, 1, core_features)
        interaction_pairs = [(i, i+1) for i in range(0, core_features-1)]
        
        # Generate source domain target
        y_source = np.zeros(n_samples)
        for i in range(core_features):
            y_source += core_coef[i] * X_source[:, i]
        
        # Add interactions
        for i, j in interaction_pairs:
            y_source += 0.5 * X_source[:, i] * X_source[:, j]
        
        # Add noise
        y_source += np.random.normal(0, 0.1, n_samples)
        
        # Target domain: shift the distribution and modify relationships
        X_target = np.random.randn(n_samples, n_features)
        
        # Make some features similar to source domain
        shared_features = int((1 - difficulty) * n_features)
        X_target[:, :shared_features] = X_source[:, :shared_features] + np.random.normal(0, 0.2, (n_samples, shared_features))
        
        # Modify coefficients for target domain
        target_coef = core_coef.copy()
        target_coef[:int(difficulty*len(target_coef))] *= np.random.uniform(0.5, 1.5, int(difficulty*len(target_coef)))
        
        # Generate target domain target with modified relationships
        y_target = np.zeros(n_samples)
        for i in range(core_features):
            y_target += target_coef[i] * X_target[:, i]
        
        # Modify some interactions
        modified_interactions = interaction_pairs.copy()
        if difficulty > 0.3:
            # Add or change some interactions based on difficulty
            if np.random.random() < difficulty:
                # Add a new interaction
                new_pair = (core_features-1, core_features)
                modified_interactions.append(new_pair)
                
        for i, j in modified_interactions:
            if i < n_features and j < n_features:  # Ensure indices are valid
                interaction_strength = 0.5
                if (i, j) not in interaction_pairs:
                    interaction_strength = 0.3  # New interactions are weaker
                y_target += interaction_strength * X_target[:, i] * X_target[:, j]
        
        # Add noise
        y_target += np.random.normal(0, 0.1, n_samples)
        
        return (X_source, y_source), (X_target, y_target)

    # Generate transfer learning data with medium difficulty
    (X_source, y_source), (X_target, y_target) = generate_transfer_data(difficulty=0.6)

    # Split datasets
    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(X_source, y_source, test_size=0.2)
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(X_target, y_target, test_size=0.2)

    # Baseline: No Transfer (train on target data only)
    def no_transfer_model(X_target_train, y_target_train):
        model = LinearRegression()
        model.fit(X_target_train, y_target_train)
        return model

    # Context-unaware: Simple Transfer (train on source, test on target)
    def simple_transfer_model(X_source_train, y_source_train):
        model = LinearRegression()
        model.fit(X_source_train, y_source_train)
        return model

    # Context-unaware: Naive Combined Transfer (train on combined data)
    def combined_transfer_model(X_source_train, y_source_train, X_target_train, y_target_train):
        X_combined = np.vstack([X_source_train, X_target_train])
        y_combined = np.concatenate([y_source_train, y_target_train])
        
        model = LinearRegression()
        model.fit(X_combined, y_combined)
        return model

    # Context-aware: Domain Adaptation Transfer
    class DomainAdaptationModel:
        def __init__(self, base_model=LinearRegression()):
            self.base_model = base_model
            self.source_model = None
            self.target_model = None
            self.feature_importance = None
            
        def fit(self, X_source, y_source, X_target, y_target):
            # Train on source domain
            self.source_model = self.base_model.__class__()
            self.source_model.fit(X_source, y_source)
            
            # Identify important features in source domain
            if hasattr(self.source_model, 'coef_'):
                self.feature_importance = np.abs(self.source_model.coef_)
            else:
                # Fallback if model doesn't expose coefficients
                self.feature_importance = np.ones(X_source.shape[1])
            
            # Normalize importance
            self.feature_importance = self.feature_importance / np.sum(self.feature_importance)
            
            # Train on target domain with source knowledge
            # Create weighted features based on source importance
            X_target_weighted = X_target * self.feature_importance
            
            # Use source predictions as additional feature
            source_preds = self.source_model.predict(X_target).reshape(-1, 1)
            X_target_augmented = np.hstack([X_target, source_preds])
            
            # Train target model with augmented features
            self.target_model = self.base_model.__class__()
            self.target_model.fit(X_target_augmented, y_target)
            
        def predict(self, X):
            # Generate predictions from source model
            source_preds = self.source_model.predict(X).reshape(-1, 1)
            
            # Combine original features with source predictions
            X_augmented = np.hstack([X, source_preds])
            
            # Use target model for final prediction
            return self.target_model.predict(X_augmented)

    # Context-aware: Feature Transformation Transfer
    class FeatureTransformTransferModel:
        def __init__(self, base_model=LinearRegression()):
            self.base_model = base_model
            self.source_model = None
            self.transformation_model = None
            self.final_model = None
            
        def fit(self, X_source, y_source, X_target, y_target):
            # Train on source domain
            self.source_model = self.base_model.__class__()
            self.source_model.fit(X_source, y_source)
            
            # Get source predictions for both domains
            source_preds_source = self.source_model.predict(X_source)
            source_preds_target = self.source_model.predict(X_target)
            
            # Learn a transformation from source predictions to target values
            self.transformation_model = LinearRegression()
            self.transformation_model.fit(source_preds_source.reshape(-1, 1), y_source)
            
            # Train final model on target domain, using transformed features
            transformed_preds = self.transformation_model.predict(source_preds_target.reshape(-1, 1))
            
            # Combine original features with transformed predictions
            X_target_augmented = np.hstack([X_target, transformed_preds.reshape(-1, 1)])
            
            self.final_model = self.base_model.__class__()
            self.final_model.fit(X_target_augmented, y_target)
            
        def predict(self, X):
            # Get source predictions
            source_preds = self.source_model.predict(X)
            
            # Transform predictions
            transformed_preds = self.transformation_model.predict(source_preds.reshape(-1, 1))
            
            # Combine with original features
            X_augmented = np.hstack([X, transformed_preds.reshape(-1, 1)])
            
            # Use final model for prediction
            return self.final_model.predict(X_augmented)

    # Evaluate transfer learning models
    def evaluate_transfer_models():
        results = []
        
        # No Transfer (baseline)
        no_transfer = no_transfer_model(X_target_train, y_target_train)
        no_transfer_mse = mean_squared_error(y_target_test, no_transfer.predict(X_target_test))
        results.append(["No Transfer (Target Only)", no_transfer_mse])
        
        # Simple Transfer
        simple_transfer = simple_transfer_model(X_source_train, y_source_train)
        simple_transfer_mse = mean_squared_error(y_target_test, simple_transfer.predict(X_target_test))
        results.append(["Simple Transfer", simple_transfer_mse])
        
        # Combined Transfer
        combined_transfer = combined_transfer_model(X_source_train, y_source_train, X_target_train, y_target_train)
        combined_transfer_mse = mean_squared_error(y_target_test, combined_transfer.predict(X_target_test))
        results.append(["Combined Transfer", combined_transfer_mse])
        
        # Domain Adaptation Transfer (Context-aware)
        da_transfer = DomainAdaptationModel()
        da_transfer.fit(X_source_train, y_source_train, X_target_train, y_target_train)
        da_transfer_mse = mean_squared_error(y_target_test, da_transfer.predict(X_target_test))
        results.append(["Domain Adaptation Transfer (CA)", da_transfer_mse])
        
        # Feature Transformation Transfer (Context-aware)
        ft_transfer = FeatureTransformTransferModel()
        ft_transfer.fit(X_source_train, y_source_train, X_target_train, y_target_train)
        ft_transfer_mse = mean_squared_error(y_target_test, ft_transfer.predict(X_target_test))
        results.append(["Feature Transformation Transfer (CA)", ft_transfer_mse])
        
        return pd.DataFrame(results, columns=["Model", "Test MSE"])

    # Run transfer learning benchmark
    transfer_results = evaluate_transfer_models()
    print("\nTransfer Learning Benchmark Results:")
    print(transfer_results)

    # Calculate improvements
    best_non_ca_transfer = min(transfer_results[transfer_results["Model"].str.contains("CA") == False]["Test MSE"])
    best_ca_transfer = min(transfer_results[transfer_results["Model"].str.contains("CA")]["Test MSE"])

    transfer_improvement = (best_non_ca_transfer - best_ca_transfer) / best_non_ca_transfer * 100
    print(f"\nImprovement from best non-CA to best CA transfer model: {transfer_improvement:.2f}%")
    
    return {
        "transfer_results": transfer_results,
        "best_non_ca_mse": best_non_ca_transfer,
        "best_ca_mse": best_ca_transfer,
        "transfer_improvement": transfer_improvement
    }

#===========================================================
# 4. Optimization Stability Benchmark
#===========================================================

def optimization_stability_benchmark():
    """Benchmark optimization algorithms with and without numerical impermanence consideration."""
    
    print("\nRunning Optimization Stability Benchmark...")
    
    # Generate ill-conditioned data where optimization is challenging
    def generate_ill_conditioned_data(n_samples=1000, n_features=20, condition_number=1000):
        # Create a random matrix
        X_raw = np.random.randn(n_samples, n_features)
        
        # Perform SVD
        U, s, Vt = np.linalg.svd(X_raw, full_matrices=False)
        
        # Modify singular values to achieve desired condition number
        s_new = np.linspace(condition_number, 1, len(s))
        
        # Reconstruct the matrix with modified singular values
        X = U @ np.diag(s_new) @ Vt
        
        # Generate true coefficients
        true_coef = np.random.uniform(-1, 1, n_features)
        
        # Generate target with noise
        y = X @ true_coef + np.random.normal(0, 0.1, n_samples)
        
        return X, y, true_coef

    # Generate the data
    X, y, true_coef = generate_ill_conditioned_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Context-unaware: Basic gradient descent
    class BasicGradientDescent:
        def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.tol = tol
            self.coef_ = None
            self.intercept_ = 0.0
            self.loss_history = []
            
        def fit(self, X, y):
            n_samples, n_features = X.shape
            
            # Initialize parameters randomly
            self.coef_ = np.random.randn(n_features) * 0.01
            
            # Gradient descent iterations
            for i in range(self.max_iter):
                # Predictions
                y_pred = X @ self.coef_ + self.intercept_
                
                # Compute loss
                loss = np.mean((y_pred - y) ** 2)
                self.loss_history.append(loss)
                
                # Check convergence
                if i > 0 and abs(self.loss_history[i] - self.loss_history[i-1]) < self.tol:
                    break
                
                # Compute gradients
                grad_coef = 2 * X.T @ (y_pred - y) / n_samples
                grad_intercept = 2 * np.mean(y_pred - y)
                
                # Update parameters
                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept
                
            return self
            
        def predict(self, X):
            return X @ self.coef_ + self.intercept_

    # Context-unaware: Basic adaptive learning rate
    class AdaptiveLearningRate:
        def __init__(self, initial_lr=0.1, decay=0.95, max_iter=1000, tol=1e-6):
            self.initial_lr = initial_lr
            self.decay = decay
            self.max_iter = max_iter
            self.tol = tol
            self.coef_ = None
            self.intercept_ = 0.0
            self.loss_history = []
            
        def fit(self, X, y):
            n_samples, n_features = X.shape
            
            # Initialize parameters randomly
            self.coef_ = np.random.randn(n_features) * 0.01
            
            # Initialize learning rate
            lr = self.initial_lr
            
            # Gradient descent iterations
            for i in range(self.max_iter):
                # Predictions
                y_pred = X @ self.coef_ + self.intercept_
                
                # Compute loss
                loss = np.mean((y_pred - y) ** 2)
                self.loss_history.append(loss)
                
                # Check convergence
                if i > 0 and abs(self.loss_history[i] - self.loss_history[i-1]) < self.tol:
                    break
                
                # Compute gradients
                grad_coef = 2 * X.T @ (y_pred - y) / n_samples
                grad_intercept = 2 * np.mean(y_pred - y)
                
                # Update parameters
                self.coef_ -= lr * grad_coef
                self.intercept_ -= lr * grad_intercept
                
                # Decay learning rate
                lr *= self.decay
                
            return self
            
        def predict(self, X):
            return X @ self.coef_ + self.intercept_

    # Context-aware: Optimization with numerical context awareness
    class ContextAwareOptimizer:
        def __init__(self, max_iter=1000, tol=1e-6):
            self.max_iter = max_iter
            self.tol = tol
            self.coef_ = None
            self.intercept_ = 0.0
            self.loss_history = []
            self.feature_scales = None
            
        def fit(self, X, y):
            n_samples, n_features = X.shape
            
            # Analyze feature scales for context-aware learning rates
            self.feature_scales = np.std(X, axis=0)
            
            # Initialize parameters randomly
            self.coef_ = np.random.randn(n_features) * 0.01
            
            # Compute initial condition and adapt optimization
            _, s, _ = np.linalg.svd(X, full_matrices=False)
            condition_number = s[0] / s[-1]
            
            # Adjust learning rates based on condition number and feature scales
            if condition_number > 100:
                # Ill-conditioned problem: use feature-specific learning rates
                base_lr = 0.01 / np.sqrt(condition_number)
                feature_lrs = base_lr / (self.feature_scales + 1e-8)
                
                # Cap learning rates to reasonable values
                feature_lrs = np.clip(feature_lrs, 1e-6, 0.1)
            else:
                # Well-conditioned problem: use uniform learning rate
                feature_lrs = np.ones(n_features) * 0.01
            
            # Gradient descent iterations with context awareness
            for i in range(self.max_iter):
                # Predictions
                y_pred = X @ self.coef_ + self.intercept_
                
                # Compute loss
                loss = np.mean((y_pred - y) ** 2)
                self.loss_history.append(loss)
                
                # Check convergence
                if i > 0 and abs(self.loss_history[i] - self.loss_history[i-1]) < self.tol:
                    break
                
                # Compute gradients
                grad_coef = 2 * X.T @ (y_pred - y) / n_samples
                grad_intercept = 2 * np.mean(y_pred - y)
                
                # Update parameters with feature-specific learning rates
                self.coef_ -= feature_lrs * grad_coef
                self.intercept_ -= 0.01 * grad_intercept
                
                # Adaptive adjustment based on recent loss trends
                if i > 5:
                    recent_losses = self.loss_history[-5:]
                    if all(x >= y for x, y in zip(recent_losses, recent_losses[1:])):
                        # Loss is consistently increasing, reduce learning rates
                        feature_lrs *= 0.5
                    elif all(x <= y for x, y in zip(recent_losses, recent_losses[1:])):
                        # Loss is consistently decreasing, consider increasing learning rates
                        feature_lrs = np.minimum(feature_lrs * 1.1, 0.1)
                
            return self
            
        def predict(self, X):
            return X @ self.coef_ + self.intercept_

    # Function to evaluate optimization methods
    def evaluate_optimizers(n_trials=10):
        results = []
        
        optimizers = [
            ("Basic GD", BasicGradientDescent()),
            ("Adaptive LR", AdaptiveLearningRate()),
            ("Context-Aware", ContextAwareOptimizer())
        ]
        
        for name, optimizer_class in optimizers:
            trial_losses = []
            trial_mses = []
            trial_iters = []
            trial_times = []
            
            for trial in range(n_trials):
                # Generate new random data for each trial
                X, y, true_coef = generate_ill_conditioned_data()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                # Train the optimizer
                start_time = time.time()
                optimizer = optimizer_class
                optimizer.fit(X_train, y_train)
                end_time = time.time()
                
                # Evaluate results
                y_pred = optimizer.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred)
                
                # Record metrics
                trial_losses.append(optimizer.loss_history[-1])
                trial_mses.append(test_mse)
                trial_iters.append(len(optimizer.loss_history))
                trial_times.append(end_time - start_time)
            
            # Compute statistics across trials
            results.append({
                "Optimizer": name,
                "Mean Test MSE": np.mean(trial_mses),
                "Std Test MSE": np.std(trial_mses),
                "Mean Iterations": np.mean(trial_iters),
                "Mean Time (s)": np.mean(trial_times),
                "Final Training Loss": np.mean(trial_losses)
            })
        
        return pd.DataFrame(results)

    # Run the optimization benchmark
    opt_results = evaluate_optimizers(n_trials=5)
    print("\nOptimization Stability Benchmark Results:")
    print(opt_results)
    
    # Calculate improvement from best non-CA to CA
    best_non_ca_opt = opt_results[opt_results["Optimizer"] != "Context-Aware"]["Mean Test MSE"].min()
    ca_opt = opt_results[opt_results["Optimizer"] == "Context-Aware"]["Mean Test MSE"].values[0]
    
    opt_improvement = (best_non_ca_opt - ca_opt) / best_non_ca_opt * 100
    
    stability_improvement = (
        opt_results[opt_results["Optimizer"] != "Context-Aware"]["Std Test MSE"].min() - 
        opt_results[opt_results["Optimizer"] == "Context-Aware"]["Std Test MSE"].values[0]
    ) / opt_results[opt_results["Optimizer"] != "Context-Aware"]["Std Test MSE"].min() * 100
    
    print(f"\nImprovement from best non-CA to Context-Aware optimizer: {opt_improvement:.2f}%")
    print(f"Stability improvement (reduction in std dev): {stability_improvement:.2f}%")
    
    return {
        "opt_results": opt_results,
        "best_non_ca_mse": best_non_ca_opt,
        "ca_mse": ca_opt,
        "opt_improvement": opt_improvement,
        "stability_improvement": stability_improvement
    }

#===========================================================
# Create summary visualizations
#===========================================================

def create_summary_plots(results):
    """Create summary plots for all benchmarks."""
    
    # Extract results
    scaling_results = results["scaling_results"]
    drift_results = results["drift_results"]
    transfer_results = results["transfer_results"]
    opt_results = results["opt_results"]
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Feature Scaling Results
    plt.subplot(2, 2, 1)
    scaling_data = pd.DataFrame({
        'Method': ['Standard', 'MinMax', 'Context-Aware'],
        'Source Domain': [scaling_results["std_mse"], scaling_results["minmax_mse"], scaling_results["ca_mse"]],
        'Target Domain': [scaling_results["std_target_mse"], scaling_results["minmax_target_mse"], scaling_results["ca_target_mse"]]
    })
    
    scaling_melted = pd.melt(scaling_data, id_vars=['Method'], var_name='Domain', value_name='MSE')
    sns.barplot(x='Method', y='MSE', hue='Domain', data=scaling_melted)
    plt.title('Feature Scaling Impact Across Domains')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')  # Log scale to better show differences
    
    # Plot 2: Concept Drift Results
    plt.subplot(2, 2, 2)
    drift_data = pd.DataFrame({
        'Method': [r["Model"] for r in drift_results["drift_results"]],
        'MSE': [r["Average MSE"] for r in drift_results["drift_results"]]
    })
    
    sns.barplot(x='Method', y='MSE', data=drift_data)
    plt.title('Model Performance Under Concept Drift')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 3: Transfer Learning Results
    plt.subplot(2, 2, 3)
    transfer_data = transfer_results["transfer_results"]
    sns.barplot(x='Model', y='Test MSE', data=transfer_data)
    plt.title('Transfer Learning Performance Comparison')
    plt.ylabel('Mean Squared Error on Target Domain')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 4: Performance Improvement Summary
    plt.subplot(2, 2, 4)
    improvements = pd.DataFrame({
        'Benchmark': ['Feature Scaling', 'Concept Drift', 'Transfer Learning', 'Optimization'],
        'Improvement (%)': [
            scaling_results["std_to_ca_improvement"],
            drift_results["drift_improvement"],
            transfer_results["transfer_improvement"],
            opt_results["opt_improvement"]
        ]
    })
    
    sns.barplot(x='Benchmark', y='Improvement (%)', data=improvements)
    plt.title('Performance Improvement from Context-Aware Approaches')
    plt.ylabel('Improvement (%)')
    
    plt.tight_layout()
    plt.savefig('numerical_impermanence_results.png')
    
    return plt

#===========================================================
# Main function
#===========================================================

def run_all_benchmarks():
    """Run all benchmarks and summarize results."""
    
    # Store all results
    all_results = {}
    
    # Run feature scaling benchmark
    scaling_results = feature_scaling_benchmark()
    all_results["scaling_results"] = scaling_results
    
    # Run concept drift benchmark
    drift_results = concept_drift_benchmark()
    all_results["drift_results"] = drift_results
    
    # Run transfer learning benchmark
    transfer_results = transfer_learning_benchmark()
    all_results["transfer_results"] = transfer_results
    
    # Run optimization stability benchmark
    opt_results = optimization_stability_benchmark()
    all_results["opt_results"] = opt_results
    
    # Create summary plots
    plt = create_summary_plots(all_results)
    
    # Print overall summary
    print("\n===== OVERALL PERFORMANCE SUMMARY =====")
    print(f"Feature Scaling Improvement: {scaling_results['std_to_ca_improvement']:.2f}%")
    print(f"Concept Drift Improvement: {drift_results['drift_improvement']:.2f}%")
    print(f"Transfer Learning Improvement: {transfer_results['transfer_improvement']:.2f}%")
    print(f"Optimization Improvement: {opt_results['opt_improvement']:.2f}%")
    print(f"Average Improvement: {np.mean([scaling_results['std_to_ca_improvement'], drift_results['drift_improvement'], transfer_results['transfer_improvement'], opt_results['opt_improvement']]):.2f}%")
    
    return all_results, plt

# Run all benchmarks if this script is executed directly
if __name__ == "__main__":
    all_results, plt = run_all_benchmarks()
    plt.show()
