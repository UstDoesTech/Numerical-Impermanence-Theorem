class ContextAwareForecaster:
    def __init__(self, base_models=None, context_types=None):
        """
        Initialize a context-aware forecasting framework
        
        Parameters:
        -----------
        base_models : dict
            Dictionary of base forecasting models by name
        context_types : list
            List of context types to detect and handle
        """
        self.base_models = base_models or {
            'trend': Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False),
            'seasonality': Prophet(growth='flat'),
            'local': ExponentialSmoothing(),
            'regime': AutoRegressive()
        }
        
        self.context_types = context_types or [
            'temporal', 'seasonal', 'regime', 'scale', 'cross_series'
        ]
        
        self.context_detectors = {}
        self.context_models = {}
        self.context_transformers = {}
        
    def detect_contexts(self, series):
        """Detect different contexts in the time series"""
        contexts = {}
        
        if 'temporal' in self.context_types:
            # Detect temporal segments (epochs, regimes)
            contexts['temporal'] = self._detect_temporal_contexts(series)
            
        if 'seasonal' in self.context_types:
            # Detect and characterize seasonality
            contexts['seasonal'] = self._detect_seasonal_contexts(series)
            
        if 'regime' in self.context_types:
            # Detect structural breaks and regime changes
            contexts['regime'] = self._detect_regime_changes(series)
            
        if 'scale' in self.context_types:
            # Detect changes in scale and volatility
            contexts['scale'] = self._detect_scale_changes(series)
            
        if 'cross_series' in self.context_types and isinstance(series, dict):
            # Analyze relationships between multiple series
            contexts['cross_series'] = self._detect_cross_series_contexts(series)
            
        return contexts
    
    def _detect_temporal_contexts(self, series):
        """Identify different temporal contexts within the series"""
        # Use change point detection to identify different epochs
        from ruptures.detection import Pelt
        
        # Convert to array if needed
        data = series.values if hasattr(series, 'values') else np.array(series)
        
        # Detect change points
        model = Pelt(model="rbf").fit(data.reshape(-1, 1))
        change_points = model.predict(pen=10)
        
        # Create temporal segments
        segments = []
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i+1]
            segments.append((start, end))
            
        return segments
    
    def _detect_seasonal_contexts(self, series):
        """Identify seasonal patterns and contexts"""
        # Extract multiple seasonal components
        from statsmodels.tsa.seasonal import STL
        
        # Check if we have enough data for seasonal decomposition
        if len(series) < 14:  # Minimum for weekly seasonality
            return {'has_seasonality': False}
            
        # Try to determine the seasonal periods
        acf_values = acf(series, nlags=len(series)//2, fft=True)
        
        # Find peaks in ACF which indicate seasonality
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acf_values, height=0.2)
        
        # If peaks are found, use them as seasonal periods
        seasonal_periods = {}
        for peak in peaks:
            if peak > 1:  # Ignore lag-1 correlation
                # Decompose with this period
                try:
                    result = STL(series, period=peak).fit()
                    # Calculate strength of seasonality
                    seasonal_strength = 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
                    seasonal_periods[peak] = seasonal_strength
                except:
                    continue
        
        return {
            'has_seasonality': len(seasonal_periods) > 0,
            'periods': seasonal_periods
        }
    
    def _detect_regime_changes(self, series):
        """Detect structural breaks and regime changes"""
        # Use Markov Switching Models to identify regime changes
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        
        try:
            # Try fitting a 2-regime model
            mod = MarkovAutoregression(series, k_regimes=2, order=1)
            res = mod.fit()
            
            # Extract regime probabilities
            smoothed_probs = res.smoothed_marginal_probabilities
            regimes = np.argmax(smoothed_probs, axis=1)
            
            # Identify transition points
            transitions = np.where(np.diff(regimes) != 0)[0] + 1
            
            return {
                'has_regimes': True,
                'regimes': regimes,
                'transitions': transitions,
                'regime_params': res.params
            }
        except:
            # Fall back to change point detection if Markov model fails
            return self._detect_temporal_contexts(series)
    
    def _detect_scale_changes(self, series):
        """Detect changes in scale and volatility"""
        # Use rolling statistics to identify scale changes
        window_size = max(30, len(series) // 10)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = series.rolling(window=window_size).mean() if hasattr(series, 'rolling') else pd.Series(series).rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std() if hasattr(series, 'rolling') else pd.Series(series).rolling(window=window_size).std()
        
        # Calculate relative changes in mean and volatility
        mean_changes = rolling_mean.pct_change(periods=window_size)
        std_changes = rolling_std.pct_change(periods=window_size)
        
        # Identify significant shifts
        significant_mean_shifts = np.where(np.abs(mean_changes) > 0.2)[0]
        significant_vol_shifts = np.where(np.abs(std_changes) > 0.3)[0]
        
        return {
            'scale_shifts': significant_mean_shifts,
            'volatility_shifts': significant_vol_shifts,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std
        }
    
    def _detect_cross_series_contexts(self, series_dict):
        """Analyze relationships between multiple series"""
        # Calculate correlations between series
        series_df = pd.DataFrame(series_dict)
        correlations = series_df.corr()
        
        # Cluster series based on correlation
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert correlation to distance (high correlation = low distance)
        distance_matrix = 1 - np.abs(correlations.values)
        np.fill_diagonal(distance_matrix, 0)
        
        # Cluster series
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=0.5,
            affinity='precomputed',
            linkage='average'
        ).fit(distance_matrix)
        
        # Extract clusters
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(series_df.columns[i])
            
        return {
            'clusters': clusters,
            'correlations': correlations
        }
    
    def transform_by_context(self, series, contexts):
        """Transform time series based on detected contexts"""
        transformations = {}
        
        # Transform based on scale context
        if 'scale' in contexts:
            scale_ctx = contexts['scale']
            
            # If we have scale shifts, apply appropriate transformations
            if len(scale_ctx['scale_shifts']) > 0:
                # Use different normalizations for different scale regimes
                transformations['scale'] = self._transform_by_scale(series, scale_ctx)
        
        # Transform based on regime context
        if 'regime' in contexts and contexts['regime'].get('has_regimes', False):
            regime_ctx = contexts['regime']
            transformations['regime'] = self._transform_by_regime(series, regime_ctx)
        
        # Transform based on seasonal context
        if 'seasonal' in contexts and contexts['seasonal'].get('has_seasonality', False):
            seasonal_ctx = contexts['seasonal']
            transformations['seasonal'] = self._transform_by_seasonality(series, seasonal_ctx)
        
        return transformations
    
    def _transform_by_scale(self, series, scale_context):
        """Apply scale-appropriate transformations"""
        # Get scale shift points
        shifts = scale_context['scale_shifts']
        
        if len(shifts) == 0:
            # If no shifts, apply global scaling
            return {'global': StandardScaler().fit_transform(series.values.reshape(-1, 1)).flatten()}
        
        # Apply different scaling to different segments
        segments = {}
        prev_shift = 0
        
        for shift in list(shifts) + [len(series)]:
            if shift - prev_shift > 10:  # Only process segments with enough data
                segment = series[prev_shift:shift]
                scaler = StandardScaler().fit(segment.values.reshape(-1, 1))
                segments[f"{prev_shift}:{shift}"] = {
                    'scaler': scaler,
                    'transformed': scaler.transform(segment.values.reshape(-1, 1)).flatten()
                }
            prev_shift = shift
            
        return segments
    
    def _transform_by_regime(self, series, regime_context):
        """Apply regime-specific transformations"""
        regimes = regime_context.get('regimes', [])
        
        if len(regimes) == 0:
            return None
            
        # Apply different models or transformations to different regimes
        regime_data = {}
        
        for regime_id in np.unique(regimes):
            # Get data for this regime
            regime_mask = regimes == regime_id
            regime_series = series[regime_mask]
            
            if len(regime_series) > 10:  # Only process regimes with enough data
                # Normalize within regime
                scaler = StandardScaler().fit(regime_series.values.reshape(-1, 1))
                
                regime_data[regime_id] = {
                    'series': regime_series,
                    'scaler': scaler,
                    'transformed': scaler.transform(regime_series.values.reshape(-1, 1)).flatten(),
                    'mean': float(regime_series.mean()),
                    'std': float(regime_series.std())
                }
                
        return regime_data
    
    def _transform_by_seasonality(self, series, seasonal_context):
        """Apply seasonality-aware transformations"""
        periods = seasonal_context.get('periods', {})
        
        if not periods:
            return None
            
        # Find the strongest seasonal period
        strongest_period = max(periods.items(), key=lambda x: x[1])[0]
        
        # Create seasonal indices
        if hasattr(series, 'index'):
            # For pandas series with datetime index
            if pd.api.types.is_datetime64_any_dtype(series.index):
                if strongest_period == 7:
                    # Weekly seasonality
                    seasonal_indices = series.index.dayofweek
                elif strongest_period in (28, 29, 30, 31):
                    # Monthly seasonality
                    seasonal_indices = series.index.day
                elif strongest_period in (365, 366):
                    # Yearly seasonality
                    seasonal_indices = series.index.dayofyear
                else:
                    # Custom seasonality
                    seasonal_indices = np.arange(len(series)) % strongest_period
            else:
                seasonal_indices = np.arange(len(series)) % strongest_period
        else:
            seasonal_indices = np.arange(len(series)) % strongest_period
            
        # Calculate seasonal means and deviations
        seasonal_stats = {}
        for idx in np.unique(seasonal_indices):
            mask = seasonal_indices == idx
            if np.sum(mask) > 0:
                seasonal_series = series[mask]
                seasonal_stats[idx] = {
                    'mean': float(seasonal_series.mean()),
                    'std': float(seasonal_series.std()),
                    'count': int(np.sum(mask))
                }
                
        # Calculate seasonally adjusted series
        seasonal_means = np.array([seasonal_stats[idx]['mean'] for idx in seasonal_indices])
        seasonally_adjusted = series.values - seasonal_means
                
        return {
            'period': strongest_period,
            'indices': seasonal_indices,
            'stats': seasonal_stats,
            'adjusted': seasonally_adjusted
        }
    
    def fit(self, series, exog=None):
        """Fit the forecasting model to the time series"""
        # Detect contexts in the series
        self.contexts = self.detect_contexts(series)
        
        # Apply context-specific transformations
        self.transformations = self.transform_by_context(series, self.contexts)
        
        # Fit specialized models for each context
        self._fit_context_models(series, exog)
        
        return self
    
    def _fit_context_models(self, series, exog=None):
        """Fit specialized models for different contexts"""
        # Prepare models dictionary
        self.models = {}
        
        # Fit global model on entire series
        global_model = self.base_models.get('trend', Prophet()).fit(
            pd.DataFrame({'ds': series.index, 'y': series.values})
        ) if hasattr(series, 'index') else LinearRegression().fit(
            np.arange(len(series)).reshape(-1, 1), series
        )
        
        self.models['global'] = global_model
        
        # Fit regime-specific models if regimes detected
        if 'regime' in self.transformations and self.transformations['regime']:
            regime_data = self.transformations['regime']
            
            for regime_id, data in regime_data.items():
                # Choose appropriate model based on regime characteristics
                if len(data['series']) >= 30:  # Enough data for time series model
                    regime_model = self.base_models.get('regime', ARIMA(order=(1,0,1))).fit(data['series'])
                else:
                    # For short regimes, use simpler model
                    regime_model = SimpleExpSmoothing(data['series']).fit()
                    
                self.models[f'regime_{regime_id}'] = regime_model
                
        # Fit seasonal models if seasonality detected
        if 'seasonal' in self.transformations and self.transformations['seasonal']:
            seasonal_data = self.transformations['seasonal']
            period = seasonal_data['period']
            
            # Fit seasonal model
            if hasattr(series, 'index'):
                seasonal_model = Prophet(yearly_seasonality=True).fit(
                    pd.DataFrame({'ds': series.index, 'y': series.values})
                )
            else:
                seasonal_model = ExponentialSmoothing(
                    series, seasonal_periods=period, seasonal='add'
                ).fit()
                
            self.models['seasonal'] = seasonal_model
        
        return self.models
    
    def predict(self, steps=10, exog=None):
        """Generate forecasts for future periods"""
        # Initial forecast from global model
        if isinstance(self.models['global'], Prophet):
            future = self.models['global'].make_future_dataframe(periods=steps)
            global_forecast = self.models['global'].predict(future)['yhat'].values[-steps:]
        else:
            global_forecast = self.models['global'].predict(
                np.arange(len(self.series), len(self.series) + steps).reshape(-1, 1)
            )
            
        # Initialize final forecast with global forecast
        final_forecast = global_forecast.copy()
        
        # Adjust based on detected regimes
        if 'regime' in self.transformations and self.transformations['regime']:
            # Determine most likely recent regime
            regime_data = self.transformations['regime']
            regime_keys = list(regime_data.keys())
            
            if regime_keys:
                # Use most recent regime for forecasting
                latest_regime = regime_keys[-1]
                regime_model = self.models.get(f'regime_{latest_regime}')
                
                if regime_model:
                    # Generate regime-specific forecast
                    try:
                        regime_forecast = regime_model.forecast(steps)
                        
                        # Blend with global forecast (more weight to regime forecast for near future)
                        weights = np.exp(-np.arange(steps) / (steps / 3))  # Decay weights
                        weights = weights / np.sum(weights)
                        
                        for i in range(steps):
                            final_forecast[i] = weights[i] * regime_forecast[i] + (1 - weights[i]) * global_forecast[i]
                    except:
                        # Fall back to global forecast if regime forecast fails
                        pass
        
        # Adjust for seasonality
        if 'seasonal' in self.transformations and self.transformations['seasonal']:
            seasonal_data = self.transformations['seasonal']
            period = seasonal_data['period']
            stats = seasonal_data['stats']
            
            # Calculate seasonal factors for future periods
            current_length = len(self.series)
            future_indices = [(current_length + i) % period for i in range(steps)]
            
            # Apply seasonal adjustments
            for i, idx in enumerate(future_indices):
                if idx in stats:
                    # Add seasonal factor to forecast
                    seasonal_factor = stats[idx]['mean'] - np.mean([s['mean'] for s in stats.values()])
                    final_forecast[i] += seasonal_factor
        
        return final_forecast