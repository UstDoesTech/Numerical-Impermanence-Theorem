import numpy as np
import matplotlib.pyplot as plt
from numerical_impermanence_framework import NumericalImpermanence, Context, apply_numerical_impermanence
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import random
import time
from decimal import Decimal, getcontext

# Set up figure style
plt.style.use('ggplot')

#=============================================================================
# Application 1: Floating-Point Arithmetic and Numerical Stability
#=============================================================================

def floating_point_demo():
    """
    Demonstrate how floating-point numbers behave differently under different contexts,
    highlighting issues with naive vs. context-aware computation.
    """
    print("=== Floating-Point Arithmetic and Numerical Impermanence ===")
    
    # Without numerical impermanence: treating floats as exact
    def without_impermanence():
        # Sum a large and a small number
        large_num = 1e16
        small_num = 1.0
        
        # Direct addition
        result = large_num + small_num
        
        print("\nWithout acknowledging numerical impermanence:")
        print(f"Adding {small_num} to {large_num:.1e}")
        print(f"Result: {result:.1f}")
        print(f"Is result > large_num? {result > large_num}")
        print(f"Is result == large_num? {result == large_num}")
        print(f"Effectively, 1.0 has been lost in the context of large magnitude")
        
        # Demonstrate catastrophic cancellation
        a = 1e8
        b = a + 1
        diff = b - a
        
        print(f"\nCatastrophic cancellation example:")
        print(f"a = {a}, b = a + 1 = {b}")
        print(f"b - a = {diff} (should be 1.0)")
        
        return result
    
    # With numerical impermanence: acknowledging the context-dependent nature of floats
    def with_impermanence():
        # Use a context-aware approach
        large_num = 1e16
        small_num = 1.0
        
        # Separate the calculations into appropriate magnitudes
        large_context = large_num
        small_context = small_num
        
        # Approach 1: Using higher precision
        getcontext().prec = 30  # Set precision for Decimal
        
        precise_large = Decimal(str(large_num))
        precise_small = Decimal(str(small_num))
        precise_result = precise_large + precise_small
        
        print("\nWith numerical impermanence (high precision context):")
        print(f"Adding {small_num} to {large_num:.1e} using high precision")
        print(f"Result: {precise_result}")
        print(f"Is result > large_num? {precise_result > precise_large}")
        
        # Approach 2: Kahan summation algorithm for context-appropriate addition
        def kahan_sum(values):
            sum = 0.0
            compensation = 0.0  # A running compensation for lost low-order bits
            
            for val in values:
                # The magic of Kahan summation: adjust the value based on previous error
                adjusted_val = val - compensation
                temp_sum = sum + adjusted_val
                # Calculate the numerical error introduced in this step
                compensation = (temp_sum - sum) - adjusted_val
                sum = temp_sum
                
            return sum
        
        # Add using Kahan summation
        kahan_result = kahan_sum([large_num, small_num])
        
        print("\nWith numerical impermanence (Kahan summation context):")
        print(f"Adding {small_num} to {large_num:.1e} using Kahan summation")
        print(f"Result: {kahan_result}")
        
        # Approach 3: Separate magnitude handling
        def magnitude_aware_sum(values):
            # Group by magnitude buckets (powers of 10)
            buckets = {}
            
            for val in values:
                if val == 0:
                    continue
                    
                magnitude = int(np.floor(np.log10(abs(val))))
                if magnitude not in buckets:
                    buckets[magnitude] = []
                buckets[magnitude].append(val)
            
            # Sum each bucket separately, then combine
            result = 0.0
            for magnitude in sorted(buckets.keys(), reverse=True):
                bucket_sum = sum(buckets[magnitude])
                result += bucket_sum
                
            return result
        
        magnitude_result = magnitude_aware_sum([large_num, small_num])
        
        print("\nWith numerical impermanence (magnitude-aware context):")
        print(f"Adding {small_num} to {large_num:.1e} using magnitude buckets")
        print(f"Result: {magnitude_result}")
        
        return precise_result, kahan_result, magnitude_result
    
    # Run both approaches
    without_result = without_impermanence()
    precise_result, kahan_result, magnitude_result = with_impermanence()
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    
    # Create comparison data
    labels = ['Standard Float', 'High Precision', 'Kahan Sum', 'Magnitude-Aware']
    diff_from_expected = [
        abs(without_result - (1e16 + 1.0)),
        abs(float(precise_result) - (1e16 + 1.0)),
        abs(kahan_result - (1e16 + 1.0)),
        abs(magnitude_result - (1e16 + 1.0))
    ]
    
    # Plot on log scale to show differences
    plt.bar(labels, diff_from_expected)
    plt.yscale('log')
    plt.title('Error in Addition (1e16 + 1.0)')
    plt.ylabel('Absolute Error (log scale)')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Error = 1.0')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    print("\nInsight from Numerical Impermanence Theorem:")
    print("Floating-point numbers don't have fixed identities - their meaning")
    print("transforms based on computational context. The same numerical value")
    print("behaves differently depending on magnitude relationships and precision context.")


#=============================================================================
# Application 2: Machine Learning Feature Normalization
#=============================================================================

def ml_normalization_demo():
    """
    Demonstrate how machine learning models can perform differently depending on
    the numerical context provided by feature normalization.
    """
    print("\n=== Machine Learning Feature Normalization and Numerical Impermanence ===")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Create features with very different scales
    X_raw = np.random.randn(n_samples, 3)  # 3 features
    X_raw[:, 0] *= 0.1                     # Small scale feature
    X_raw[:, 1] *= 1000                    # Large scale feature
    X_raw[:, 2] *= 1                       # Medium scale feature
    
    # True relationship: y = 5*X0 + 0.001*X1 + 1*X2 + noise
    # Note that the second coefficient is small because X1 is large
    true_coef = np.array([5, 0.001, 1])
    y = np.dot(X_raw, true_coef) + np.random.randn(n_samples) * 0.1
    
    # Without numerical impermanence: ignoring context of feature scales
    def without_impermanence():
        # Train model directly on raw features
        model = LinearRegression()
        model.fit(X_raw, y)
        
        pred = model.predict(X_raw)
        mse = np.mean((pred - y) ** 2)
        
        print("\nWithout acknowledging numerical impermanence:")
        print("Training with raw features (ignoring scale differences)")
        print(f"True coefficients: {true_coef}")
        print(f"Learned coefficients: {model.coef_}")
        print(f"Mean Squared Error: {mse:.6f}")
        
        # See how model behaves under slight perturbations to features
        perturbed_X = X_raw.copy()
        perturbed_X[:, 0] += 0.01  # Small perturbation to small-scale feature
        perturbed_X[:, 1] += 10    # Larger perturbation to large-scale feature
        
        perturbed_pred = model.predict(perturbed_X)
        perturbed_mse = np.mean((perturbed_pred - y) ** 2)
        
        print(f"MSE after small perturbation: {perturbed_mse:.6f}")
        print(f"Perturbation sensitivity: {(perturbed_mse - mse) / mse:.2%}")
        
        return model.coef_, mse, perturbed_mse
    
    # With numerical impermanence: acknowledging context via normalization
    def with_impermanence():
        # Standardize features to create a normalized context
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        # Train model on standardized features
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Transform coefficients back to original scale for comparison
        original_scale_coef = model.coef_ / scaler.scale_
        
        pred = model.predict(X_scaled)
        mse = np.mean((pred - y) ** 2)
        
        print("\nWith numerical impermanence (normalization context):")
        print("Training with standardized features (scale-aware context)")
        print(f"True coefficients: {true_coef}")
        print(f"Learned coefficients (original scale): {original_scale_coef}")
        print(f"Mean Squared Error: {mse:.6f}")
        
        # Apply the same perturbation as before, but in normalized space
        perturbed_X_raw = X_raw.copy()
        perturbed_X_raw[:, 0] += 0.01
        perturbed_X_raw[:, 1] += 10
        
        # Apply the same scaler
        perturbed_X_scaled = scaler.transform(perturbed_X_raw)
        
        perturbed_pred = model.predict(perturbed_X_scaled)
        perturbed_mse = np.mean((perturbed_pred - y) ** 2)
        
        print(f"MSE after small perturbation: {perturbed_mse:.6f}")
        print(f"Perturbation sensitivity: {(perturbed_mse - mse) / mse:.2%}")
        
        return original_scale_coef, mse, perturbed_mse
    
    # Run both approaches
    raw_coef, raw_mse, raw_perturbed_mse = without_impermanence()
    norm_coef, norm_mse, norm_perturbed_mse = with_impermanence()
    
    # Visualize the results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Coefficients comparison
    plt.subplot(2, 1, 1)
    coef_data = pd.DataFrame({
        'True': true_coef,
        'Raw': raw_coef,
        'Normalized': norm_coef
    }, index=['Feature 1 (small)', 'Feature 2 (large)', 'Feature 3 (medium)'])
    
    coef_data.plot(kind='bar', ax=plt.gca())
    plt.title('Coefficient Comparison')
    plt.ylabel('Coefficient Value')
    plt.legend(title='Approach')
    
    # Plot 2: MSE before and after perturbation
    plt.subplot(2, 1, 2)
    mse_data = pd.DataFrame({
        'Original': [raw_mse, norm_mse],
        'After Perturbation': [raw_perturbed_mse, norm_perturbed_mse]
    }, index=['Raw Features', 'Normalized Features'])
    
    mse_data.plot(kind='bar', ax=plt.gca())
    plt.title('Model Sensitivity to Perturbations')
    plt.ylabel('Mean Squared Error')
    plt.legend(title='Data State')
    
    plt.tight_layout()
    plt.show()
    
    print("\nInsight from Numerical Impermanence Theorem:")
    print("Machine learning features don't have fixed meanings or importances.")
    print("Their identities transform based on the normalization context, changing")
    print("how the model interprets and utilizes them. The same numerical value")
    print("can be significant in one context and negligible in another.")


#=============================================================================
# Application 3: Financial Risk Assessment
#=============================================================================

def financial_risk_demo():
    """
    Demonstrate how financial risk assessment changes depending on the temporal
    and market context of numerical values.
    """
    print("\n=== Financial Risk Assessment and Numerical Impermanence ===")
    
    # Generate synthetic stock price data
    np.random.seed(42)
    days = 1000
    
    # Base price movement
    price_changes = np.random.normal(0.0005, 0.01, days)
    
    # Add regime changes - periods of different volatility
    regimes = [
        (0, 250, 1.0),      # Normal volatility
        (250, 500, 2.5),    # High volatility (e.g., market crisis)
        (500, 750, 0.5),    # Low volatility (e.g., stable growth)
        (750, 1000, 1.2)    # Return to moderate volatility
    ]
    
    for start, end, vol_factor in regimes:
        price_changes[start:end] *= vol_factor
    
    # Calculate prices from changes (starting at 100)
    prices = 100 * np.cumprod(1 + price_changes)
    
    # Calculate returns
    returns = price_changes * 100  # Convert to percentage
    
    # Without numerical impermanence: static risk model
    def without_impermanence():
        # Use a fixed value-at-risk calculation regardless of market regime
        confidence_level = 0.95
        lookback_window = 60  # Rolling 60-day window
        
        # Calculate rolling VaR using a fixed window
        var_values = []
        
        for i in range(lookback_window, len(returns)):
            historical_returns = returns[i-lookback_window:i]
            var_95 = np.percentile(historical_returns, (1-confidence_level)*100)
            var_values.append(var_95)
        
        # Add zeros for the initial period where we don't have enough data
        var_values = [0] * lookback_window + var_values
        
        print("\nWithout acknowledging numerical impermanence:")
        print("Using static risk model with fixed lookback window")
        print(f"95% Value at Risk at day 300 (high volatility): {var_values[300]:.2f}%")
        print(f"95% Value at Risk at day 600 (low volatility): {var_values[600]:.2f}%")
        
        # Calculate how often the VaR is breached (returns lower than VaR)
        breaches = sum(1 for i in range(lookback_window, len(returns)) 
                      if returns[i] < var_values[i])
        expected_breaches = (1 - confidence_level) * (len(returns) - lookback_window)
        
        print(f"Expected VaR breaches: {expected_breaches:.1f}")
        print(f"Actual VaR breaches: {breaches}")
        print(f"Breach ratio: {breaches / expected_breaches:.2f}x expected")
        
        return var_values, breaches / expected_breaches
    
    # With numerical impermanence: adaptive, context-aware risk model
    def with_impermanence():
        confidence_level = 0.95
        
        # Adaptive approach - window size and model adjusts based on volatility
        var_values = []
        realized_vol = []
        
        # First, calculate realized volatility
        vol_window = 30
        for i in range(vol_window, len(returns)):
            vol = np.std(returns[i-vol_window:i]) * np.sqrt(252)  # Annualized
            realized_vol.append(vol)
        
        # Add zeros for the initial period
        realized_vol = [0] * vol_window + realized_vol
        
        # Now calculate adaptive VaR
        for i in range(vol_window, len(returns)):
            # Adjust lookback window based on volatility regime
            # Higher volatility → shorter window to react faster
            # Lower volatility → longer window for more stability
            current_vol = realized_vol[i]
            
            if current_vol > 25:  # High volatility regime
                adaptive_window = 30  # Shorter window
                # Use heavier tails in high volatility
                t_factor = 1.2
            elif current_vol < 10:  # Low volatility regime
                adaptive_window = 90  # Longer window
                t_factor = 0.9
            else:  # Medium volatility regime
                adaptive_window = 60  # Medium window
                t_factor = 1.0
            
            # Ensure we have enough data
            effective_window = min(adaptive_window, i)
            
            # Calculate VaR with adjustment for non-normality
            historical_returns = returns[i-effective_window:i]
            var_95 = np.percentile(historical_returns, (1-confidence_level)*100) * t_factor
            var_values.append(var_95)
        
        # Add zeros for the initial period
        var_values = [0] * vol_window + var_values
        
        print("\nWith numerical impermanence (adaptive context):")
        print("Using adaptive risk model with context-dependent parameters")
        print(f"95% Value at Risk at day 300 (high volatility): {var_values[300]:.2f}%")
        print(f"95% Value at Risk at day 600 (low volatility): {var_values[600]:.2f}%")
        
        # Calculate breaches
        breaches = sum(1 for i in range(vol_window, len(returns)) 
                      if returns[i] < var_values[i])
        expected_breaches = (1 - confidence_level) * (len(returns) - vol_window)
        
        print(f"Expected VaR breaches: {expected_breaches:.1f}")
        print(f"Actual VaR breaches: {breaches}")
        print(f"Breach ratio: {breaches / expected_breaches:.2f}x expected")
        
        return var_values, realized_vol, breaches / expected_breaches
    
    # Run both approaches
    static_var, static_breach_ratio = without_impermanence()
    adaptive_var, realized_vol, adaptive_breach_ratio = with_impermanence()
    
    # Visualize the results
    plt.figure(figsize=(12, 12))
    
    # Plot 1: Price data and volatility regimes
    plt.subplot(3, 1, 1)
    plt.plot(prices)
    
    # Add shaded regions for different regimes
    for start, end, vol_factor in regimes:
        plt.axvspan(start, end, alpha=0.2, 
                   color='green' if vol_factor < 1 else ('red' if vol_factor > 1.5 else 'blue'))
    
    plt.title('Synthetic Stock Price with Volatility Regimes')
    plt.ylabel('Price')
    
    # Add a legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.2, label='Normal Volatility'),
        Patch(facecolor='red', alpha=0.2, label='High Volatility'),
        Patch(facecolor='green', alpha=0.2, label='Low Volatility')
    ]
    plt.legend(handles=legend_elements)
    
    # Plot 2: Returns and VaR models
    plt.subplot(3, 1, 2)
    plt.plot(returns, color='gray', alpha=0.5, label='Daily Returns')
    plt.plot(static_var, 'r-', label='Static VaR (95%)')
    plt.plot(adaptive_var, 'g-', label='Adaptive VaR (95%)')
    
    plt.title('Returns and Value-at-Risk Models')
    plt.ylabel('Daily Return (%)')
    plt.legend()
    
    # Plot 3: Comparison of VaR models across different regimes
    plt.subplot(3, 1, 3)
    
    # Extract VaR values for each regime
    regime_centers = [125, 375, 625, 875]  # Center of each regime
    static_regime_vars = [static_var[i] for i in regime_centers]
    adaptive_regime_vars = [adaptive_var[i] for i in regime_centers]
    
    # Calculate average volatility in each regime
    regime_vols = []
    for start, end, _ in regimes:
        regime_vols.append(np.std(returns[start:end]) * np.sqrt(252))
    
    # Plot the comparison
    width = 0.35
    x = np.arange(len(regimes))
    
    plt.bar(x - width/2, static_regime_vars, width, label='Static VaR')
    plt.bar(x + width/2, adaptive_regime_vars, width, label='Adaptive VaR')
    
    plt.title('VaR Comparison Across Market Regimes')
    plt.xlabel('Market Regime')
    plt.ylabel('Value at Risk (%)')
    plt.xticks(x, ['Normal', 'High Vol', 'Low Vol', 'Moderate'])
    
    # Add a second y-axis showing the volatility
    ax2 = plt.gca().twinx()
    ax2.plot(x, regime_vols, 'ro-', label='Realized Volatility')
    ax2.set_ylabel('Annualized Volatility (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    print("\nInsight from Numerical Impermanence Theorem:")
    print("Risk measures don't have fixed meanings across different market regimes.")
    print("The same historical return value has different implications in different")
    print("volatility contexts. By acknowledging the temporal transformation of")
    print("numerical meaning, adaptive models better capture true financial risk.")


#=============================================================================
# Application 4: Quantum Computing Simulation
#=============================================================================

def quantum_computing_demo():
    """
    Demonstrate how classical vs. quantum computation embodies the principle of 
    numerical impermanence.
    """
    print("\n=== Quantum Computing and Numerical Impermanence ===")
    
    # Define the problem: searching an unsorted database
    database_size = 1024
    target_index = random.randrange(database_size)
    
    print(f"Problem: Find target item at index {target_index} in an unsorted database of size {database_size}")
    
    # Without numerical impermanence: classical search
    def without_impermanence():
        # Classical search - must check elements sequentially
        found_at = -1
        comparisons = 0
        
        start_time = time.time()
        
        # Simulate classical search
        for i in range(database_size):
            comparisons += 1
            if i == target_index:
                found_at = i
                break
        
        end_time = time.time()
        search_time = end_time - start_time
        
        print("\nWithout acknowledging numerical impermanence (classical context):")
        print(f"Found target at index {found_at} after {comparisons} comparisons")
        print(f"Search time: {search_time:.6f} seconds")
        
        # Complexity analysis
        avg_comparisons = database_size / 2  # Average case
        worst_comparisons = database_size    # Worst case
        
        print(f"Average case complexity: O(n) = {avg_comparisons} comparisons")
        print(f"Worst case complexity: O(n) = {worst_comparisons} comparisons")
        
        return comparisons, search_time
    
    # With numerical impermanence: quantum search simulation
    def with_impermanence():
        # Simulate Grover's algorithm behavior
        # In real quantum computing, we'd implement this on quantum hardware
        
        start_time = time.time()
        
        # Grover's algorithm requires approximately π/4 * sqrt(N) iterations
        iterations = int(np.pi/4 * np.sqrt(database_size))
        
        # Simulate the behavior of quantum operations
        # In each iteration, we increase the amplitude of the target state
        probability_target = 0.0
        
        for i in range(iterations):
            # After each iteration, probability of measuring target increases
            probability_target = np.sin((2*i + 1) * np.arcsin(1/np.sqrt(database_size))) ** 2
        
        # Simulate final measurement (would collapse the quantum state)
        # With high probability, we measure the target state
        found = (random.random() < probability_target)
        found_at = target_index if found else random.randrange(database_size)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        print("\nWith numerical impermanence (quantum context):")
        print(f"Used Grover's algorithm with {iterations} iterations")
        print(f"Final probability of finding target: {probability_target:.6f}")
        print(f"Measurement result: {'Success' if found else 'Failure'}, found index {found_at}")
        print(f"Simulation time: {search_time:.6f} seconds")
        
        # Complexity analysis
        quantum_complexity = np.pi/4 * np.sqrt(database_size)
        speedup_factor = database_size / (2 * quantum_complexity)
        
        print(f"Quantum complexity: O(√n) ≈ {quantum_complexity:.1f} iterations")
        print(f"Quantum speedup: ~{speedup_factor:.1f}x faster than classical search")
        
        return iterations, probability_target, search_time
    
    # Run both approaches
    classical_comparisons, classical_time = without_impermanence()
    quantum_iterations, quantum_prob, quantum_time = with_impermanence()
    
    # Visualize the comparison for different database sizes
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Complexity comparison
    plt.subplot(2, 1, 1)
    sizes = np.arange(100, 10001, 100)
    classical_ops = sizes / 2  # Average case
    quantum_ops = np.pi/4 * np.sqrt(sizes)
    
    plt.plot(sizes, classical_ops, 'r-', label='Classical Search O(n)')
    plt.plot(sizes, quantum_ops, 'b-', label='Quantum Search O(√n)')
    plt.axvline(x=database_size, color='g', linestyle='--', 
                label=f'Current size: {database_size}')
    
    plt.title('Search Algorithm Complexity Comparison')
    plt.xlabel('Database Size')
    plt.ylabel('Number of Operations')
    plt.legend()
    
    # Plot 2: Quantum state evolution in Grover's algorithm
    plt.subplot(2, 1, 2)
    max_iterations = int(np.pi/4 * np.sqrt(database_size)) * 2
    iterations = np.arange(max_iterations)
    
    # Calculate probability of measuring target state after each iteration
    probabilities = np.sin((2*iterations + 1) * np.arcsin(1/np.sqrt(database_size))) ** 2
    
    plt.plot(iterations, probabilities, 'b-')
    plt.axhline(y=1.0, color='g', linestyle='--', label='Perfect probability')
    plt.axvline(x=quantum_iterations, color='r', linestyle='--', 
                label=f'Optimal iterations: {quantum_iterations}')
    
    plt.title('Quantum State Evolution in Grover\'s Algorithm')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measuring Target State')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nInsight from Numerical Impermanence Theorem:")
    print("In quantum computing, numbers exist in superposition across multiple states.")
    print("The value '1' can simultaneously be partially present in multiple locations.")
    print("This fundamentally different context transforms how we process and search")
    print("for information, enabling quadratic speedups for search problems.")


#=============================================================================
# Application 5: Cross-Cultural Number Systems
#=============================================================================

def cross_cultural_numbers_demo():
    """
    Demonstrate how numerical representation and meaning varies across different
    cultural and linguistic contexts.
    """
    print("\n=== Cross-Cultural Number Systems and Numerical Impermanence ===")
    
    # Example number for demonstration
    number = 12
    
    # Without numerical impermanence: fixed decimal representation
    def without_impermanence():
        print("\nWithout acknowledging numerical impermanence (fixed decimal context):")
        print(f"Number: {number}")
        print(f"Western decimal representation: {number}")
        print(f"Binary representation: {bin(number)}")
        print(f"Hexadecimal representation: {hex(number)}")
        
        # Simple operations
        print(f"Double: {number * 2}")
        print(f"Square: {number ** 2}")
        
        # Display as a point on a number line
        plt.figure(figsize=(10, 2))
        plt.plot([0, 20], [0, 0], 'k-', alpha=0.3)  # Number line
        plt.plot([number], [0], 'ro', markersize=10)  # Mark the number
        
        # Add tick marks
        plt.xticks(range(0, 21))
        plt.yticks([])
        
        plt.title(f'Decimal Number Line Representation of {number}')
        plt.tight_layout()
        plt.show()
        
        return number
    
    # With numerical impermanence: contextual representation
    def with_impermanence():
        print("\nWith numerical impermanence (multi-cultural context):")
        
        # Different base systems
        bases = {
            'Binary': 2,
            'Octal': 8,
            'Decimal': 10,
            'Dozenal': 12,
            'Hexadecimal': 16,
            'Vigesimal (Maya/Aztec)': 20,
            'Sexagesimal (Babylonian)': 60
        }
        
        # Convert to different bases
        base_representations = {}
        for name, base in bases.items():
            if base <= 36:  # Standard conversion works for bases up to 36
                digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                representation = ""
                n = number
                
                while n > 0:
                    representation = digits[n % base] + representation
                    n //= base
                
                if representation == "":
                    representation = "0"
                    
                base_representations[name] = representation
            else:
                # For larger bases, use a different approach
                n = number
                representation = []
                
                while n > 0:
                    representation.insert(0, n % base)
                    n //= base
                
                if not representation:
                    representation = [0]
                    
                base_representations[name] = representation
        
        # Display the representations
        for name, rep in base_representations.items():
            print(f"{name} representation: {rep}")
        
        # Cultural calendar systems
        calendars = {
            'Gregorian Year': 2023,
            'Chinese Zodiac': 'Rabbit',
            'Islamic Year': 1444,
            'Hebrew Year': 5783,
            'Hindu Vikram Samvat': 2079
        }
        
        print("\nCultural calendar contexts:")
        for name, year in calendars.items():
            print(f"{name}: {year}")
        
        # Numerical context in language
        languages = {
            'English': 'twelve',
            'Spanish': 'doce',
            'French': 'douze',
            'German': 'zwölf',
            'Chinese': '十二 (shí'èr, lit. "ten-two")',
            'Japanese': '十二 (jūni, lit. "ten-two")',
            'Hindi': 'बारह (bārah)'
        }
        
        print("\nLinguistic representations:")
        for lang, word in languages.items():
            print(f"{lang}: {word}")
        
        # Visualize different representations
        plt.figure(figsize=(12, 8))
        
        # Base system comparison
        plt.subplot(2, 1, 1)
        base_names = list(bases.keys())
        
        # Convert representations to display length
        display_lengths = []
        for name in base_names:
            rep = base_representations[name]
            if isinstance(rep, list):
                length = len(rep)
            else:
                length = len(rep)
            display_lengths.append(length)
        
        plt.bar(base_names, display_lengths)
        plt.title(f'Representation Length of {number} in Different Base Systems')
        plt.ylabel('Digits Required')
        plt.xticks(rotation=45, ha='right')
        
        # Language representation comparison
        plt.subplot(2, 1, 2)
        lang_names = list(languages.keys())
        word_lengths = [len(word.split('(')[0].strip()) for word in languages.values()]
        
        plt.bar(lang_names, word_lengths)
        plt.title(f'Word Length for {number} in Different Languages')
        plt.ylabel('Characters in Word')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        return base_representations, languages
    
    # Run both approaches
    decimal_rep = without_impermanence()
    base_reps, lang_reps = with_impermanence()
    
    print("\nInsight from Numerical Impermanence Theorem:")
    print("Numbers have no fixed representation across cultures and languages.")
    print("The same quantity can be represented in numerous ways, each embedding")
    print("different conceptual frameworks. Base systems reflect historical counting")
    print("methods (fingers, lunar cycles, etc.), while linguistic representations")
    print("reveal how different cultures conceptualize quantity.")


#=============================================================================
# Main function to run all demos
#=============================================================================

def run_all_demos():
    """Run all application demos sequentially."""
    floating_point_demo()
    ml_normalization_demo()
    financial_risk_demo()
    quantum_computing_demo()
    cross_cultural_numbers_demo()

# Run the demos if this script is executed directly
if __name__ == "__main__":
    run_all_demos()
