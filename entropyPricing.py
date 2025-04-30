from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime
import os
import pickle
import datetime

# File path constants for caching
CACHE_DIR = "cache"
COMPARISON_RESULTS_FILE = os.path.join(CACHE_DIR, "option_comparison_results.pkl")
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "iv_model.pkl")

# Data parameters
DATA_START_DATE = '2020-01-01'  # Start date for data fetching
DATA_END_DATE = '2024-12-31'    # End date for data fetching
ENABLE_PLOT = True              # Set to True to enable visualization

# Entropy estimation parameters
ENTROPY_BINS = 10               # Number of bins for discretization in entropy calculation
ENTROPY_REGULARIZATION = None   # Regularization parameter (None for automatic calculation)
MIN_SAMPLES_FOR_CORRELATION = 3 # Minimum samples required for correlation calculation

# Rolling window parameters
ROLLING_WINDOW_SIZES = [5, 10, 20, 30, 60]  # Window sizes for rolling entropy calculation
DEFAULT_WINDOW_SIZE = 20         # Default window size for visualization

# Lead-lag analysis parameters
MAX_LAG_DAYS = 10               # Maximum days to test for lead-lag relationships
LAG_STEP = 1                    # Step size between lag tests

# Trading strategy parameters
ENTROPY_THRESHOLD = 0.7         # Threshold for trading signals (normalized entropy)

# Risk-free rate (can be adjusted based on time period)
RISK_FREE_RATE = 0.03

def fetch_data(start=DATA_START_DATE, end=DATA_END_DATE):
    sp500 = yf.download("^GSPC", start=start, end=end)
    vix = yf.download("^VIX", start=start, end=end)
    spy = yf.download("SPY", start=start, end=end)
    #convert to pd
    sp500 = pd.DataFrame(sp500)
    vix = pd.DataFrame(vix)
    spy = pd.DataFrame(spy)
    return sp500, vix, spy

# Define the Paninski entropy estimator class
class PaninskiEntropyEstimator:
    def __init__(self, bins=None, regularization=None):
        self.bins = bins
        self.regularization = regularization
    
    def _discretize(self, data):
        """Discretize continuous data into bins"""
        if self.bins is None:
            # Use Scott's rule for bin determination
            bin_width = 3.5 * np.std(data) / (len(data) ** (1/3))
            self.bins = max(int((max(data) - min(data)) / bin_width), 2)
        
        hist, bin_edges = np.histogram(data, bins=self.bins)
        binned_data = np.digitize(data, bin_edges[:-1])
        return binned_data, self.bins
    
    def _optimal_regularization(self, n, k):
        """Determine optimal regularization parameter"""
        if n >= k:
            return 0.5
        else:
            return max(0.5, (k - n) / (2 * n))
    
    def estimate_entropy(self, data):
        """Estimate entropy using Paninski's method"""
        # Discretize the data
        discretized_data, alphabet_size = self._discretize(data)
        
        # Count occurrences of each symbol
        unique_symbols, counts = np.unique(discretized_data, return_counts=True)
        n = len(discretized_data)
        
        # Get or compute the regularization parameter
        if self.regularization is None:
            self.regularization = self._optimal_regularization(n, alphabet_size)
        
        # Initialize probability estimates for all possible symbols
        estimated_probs = np.zeros(alphabet_size)
        
        # Set probabilities based on observed counts with regularization
        for i, symbol in enumerate(unique_symbols):
            estimated_probs[symbol-1] = (counts[i] + self.regularization) / (n + alphabet_size * self.regularization)
        
        # Handle unobserved symbols
        zero_indices = np.where(estimated_probs == 0)[0]
        for idx in zero_indices:
            estimated_probs[idx] = self.regularization / (n + alphabet_size * self.regularization)
        
        # Normalize probabilities to ensure they sum to 1
        estimated_probs = estimated_probs / np.sum(estimated_probs)
        
        # Calculate entropy using the estimated probabilities
        entropy = 0
        for p in estimated_probs:
            if p > 0:  # Avoid log(0)
                entropy -= p * np.log2(p)
        
        return entropy

# Calculate rolling returns for S&P 500
def calculate_returns(df, window=1):
    """Calculate log returns from price data"""
    # Use Close for return calculation
    returns = np.log(df['Close'] / df['Close'].shift(window)).dropna()
    return returns

# Calculate rolling entropy
def calculate_rolling_entropy(returns, window_size=20, step=1):
    """Calculate rolling entropy values using the Paninski estimator"""
    estimator = PaninskiEntropyEstimator(bins=10)  # Use fixed bins for consistency
    entropy_values = []
    dates = []
    
    for i in range(0, len(returns) - window_size + 1, step):
        window_data = returns.iloc[i:i+window_size]
        entropy = estimator.estimate_entropy(window_data.values)
        entropy_values.append(entropy)
        dates.append(returns.index[i+window_size-1])
    
    return pd.Series(entropy_values, index=dates)

# Normalize the entropy and VIX for comparison
def normalize_series(series):
    """Normalize series to 0-1 range for easier comparison"""
    normalized = (series - series.min()) / (series.max() - series.min())
    return normalized

# Calculate realized volatility
def realized_volatility(returns, window=20):
    """Calculate realized volatility (annualized)"""
    return returns.rolling(window).std() * np.sqrt(252)


# Enhanced function to calibrate implied volatility using both entropy and historical volatility
def calibrate_implied_volatility(entropy_series, hist_vol_series, vix_series):
    """
    Create a model that maps entropy and historical volatility to implied volatility (VIX)
    
    Parameters:
    - entropy_series: Series of entropy values
    - hist_vol_series: Series of historical volatility values
    - vix_series: Series of VIX values (proxy for implied volatility)
    
    Returns:
    - Calibrated regression model and evaluation metrics
    """
    # Align all series
    common_idx = entropy_series.index.intersection(
        hist_vol_series.index.intersection(vix_series.index)
    )
    
    if len(common_idx) < 10:
        raise ValueError("Insufficient data for calibration (less than 10 common dates)")
    
    # Print debug info about our data shapes
    print(f"Entropy series shape: {entropy_series.loc[common_idx].shape}")
    print(f"Hist vol series shape: {hist_vol_series.loc[common_idx].shape}")
    print(f"VIX series shape: {vix_series.loc[common_idx].shape}")
    
    # Make sure everything is 1D by forcing to numpy arrays and flattening
    entropy_values = np.array(entropy_series.loc[common_idx]).flatten()
    hist_vol_values = np.array(hist_vol_series.loc[common_idx]).flatten()
    vix_values = np.array(vix_series.loc[common_idx]).flatten()
    
    # Prepare features (X) and target (y) with these flattened arrays
    X = pd.DataFrame({
        'entropy': entropy_values,
        'hist_vol': hist_vol_values
    })
    y = vix_values
    
    # Print shapes after flattening
    print(f"X shape after flattening: {X.shape}")
    print(f"y shape after flattening: {y.shape}")
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict and evaluate
    y_pred = model.predict(X)
    
    # Calculate evaluation metrics
    metrics = {
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'coefficients': {
            'intercept': model.intercept_,
            'entropy': model.coef_[0],
            'hist_vol': model.coef_[1]
        }
    }
    
    return model, metrics

# Function to predict implied volatility using the calibrated model - FIXED VERSION
def predict_implied_volatility(model, entropy, hist_vol):
    """
    Predict implied volatility using the calibrated model
    
    Parameters:
    - model: Calibrated regression model
    - entropy: Entropy value or series
    - hist_vol: Historical volatility value or series
    
    Returns:
    - Predicted implied volatility value or series
    """
    if isinstance(entropy, pd.Series) and isinstance(hist_vol, pd.Series):
        # Align dates
        common_idx = entropy.index.intersection(hist_vol.index)
        
        # Ensure we have flattened arrays
        entropy_values = np.array(entropy.loc[common_idx]).flatten()
        hist_vol_values = np.array(hist_vol.loc[common_idx]).flatten()
        
        # Use the same feature names as during training
        X = pd.DataFrame({
            'entropy': entropy_values,
            'hist_vol': hist_vol_values
        })
        pred = model.predict(X)
        return pd.Series(pred, index=common_idx)
    else:
        # Handle single value case - properly extract from Series if needed
        if isinstance(entropy, pd.Series):
            entropy_val = float(entropy.iloc[0])
        else:
            entropy_val = float(entropy)
            
        if isinstance(hist_vol, pd.Series):
            hist_vol_val = float(hist_vol.iloc[0])
        else:
            hist_vol_val = float(hist_vol)
        
        # Use the same feature names as during training
        X = pd.DataFrame({
            'entropy': [entropy_val],
            'hist_vol': [hist_vol_val]
        })
        return model.predict(X)[0]# Black-Scholes option pricing functions
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a call option
    
    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility
    
    Returns:
    - Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a put option
    
    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility
    
    Returns:
    - Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put_price

# Function to process SPY options data
def process_options_data(file_path):
    """
    Process SPY options data from CSV file
    
    Parameters:
    - file_path: Path to the CSV file containing options data
    
    Returns:
    - Processed DataFrame with options data
    """
    try:
        # First, read a few lines to examine the format
        with open(file_path, 'r') as f:
            sample = ''.join([f.readline() for _ in range(5)])
        print(f"Sample of options file:\n{sample}")
        
        # Read CSV file with better error handling
        options_df = pd.read_csv(file_path, sep=',', low_memory=False)
        print(f"Original columns: {options_df.columns.tolist()}")
        
        # Handle brackets in column names
        columns = {}
        for col in options_df.columns:
            # Remove brackets and trim
            clean_col = col.replace('[', '').replace(']', '').strip()
            columns[col] = clean_col
        
        # Rename columns
        options_df = options_df.rename(columns=columns)
        print(f"Cleaned columns: {options_df.columns.tolist()}")
        
        # Convert numeric columns to appropriate types
        numeric_cols = [
            'UNDERLYING_LAST', 'DTE', 'STRIKE', 
            'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV', 
            'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK',
            'P_BID', 'P_ASK', 'P_LAST', 'P_DELTA', 'P_GAMMA', 'P_VEGA', 
            'P_THETA', 'P_RHO', 'P_IV', 'P_VOLUME', 
            'STRIKE_DISTANCE', 'STRIKE_DISTANCE_PCT'
        ]
        
        for col in numeric_cols:
            if col in options_df.columns:
                # First, replace empty strings with NaN
                options_df[col] = options_df[col].replace('', np.nan)
                # Then convert to float, coercing errors to NaN
                options_df[col] = pd.to_numeric(options_df[col], errors='coerce')
        
        # Convert date columns to datetime
        options_df['QUOTE_DATE'] = pd.to_datetime(options_df['QUOTE_DATE'])
        options_df['EXPIRE_DATE'] = pd.to_datetime(options_df['EXPIRE_DATE'])
        
        # Calculate mid prices from bid and ask
        if 'C_BID' in options_df.columns and 'C_ASK' in options_df.columns:
            options_df['C_MID'] = (options_df['C_BID'] + options_df['C_ASK']) / 2
        
        if 'P_BID' in options_df.columns and 'P_ASK' in options_df.columns:
            options_df['P_MID'] = (options_df['P_BID'] + options_df['P_ASK']) / 2
        
        # Filter options with non-zero prices and reasonable IVs
        filter_conditions = []
        
        if 'C_MID' in options_df.columns:
            filter_conditions.append(options_df['C_MID'] > 0)
        if 'P_MID' in options_df.columns:
            filter_conditions.append(options_df['P_MID'] > 0)
        if 'C_IV' in options_df.columns:
            filter_conditions.append(options_df['C_IV'] > 0)
            filter_conditions.append(options_df['C_IV'] < 2)
        if 'P_IV' in options_df.columns:
            filter_conditions.append(options_df['P_IV'] > 0)
            filter_conditions.append(options_df['P_IV'] < 2)
        
        if filter_conditions:
            # Use all conditions with AND logic
            final_filter = filter_conditions[0]
            for cond in filter_conditions[1:]:
                final_filter = final_filter & cond
                
            options_df = options_df[final_filter]
        
        print(f"Processed {len(options_df)} option records")
        return options_df
        
    except Exception as e:
        print(f"Error processing options data: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to compare model prices with market prices
def compare_option_prices(options_df, iv_model, entropy_series, hist_vol_series, risk_free_rate=RISK_FREE_RATE):
    """
    Compare model-derived option prices with market prices
    
    Parameters:
    - options_df: DataFrame containing SPY options data
    - iv_model: Calibrated implied volatility model
    - entropy_series: Series of entropy values
    - hist_vol_series: Series of historical volatility values
    - risk_free_rate: Risk-free interest rate
    
    Returns:
    - DataFrame with comparison results
    """
    results = []
    
    # Get unique dates in options data - use the column name we know exists
    if 'QUOTE_DATE' not in options_df.columns:
        print(f"Column 'QUOTE_DATE' not found. Available columns: {options_df.columns.tolist()}")
        return pd.DataFrame()
        
    dates = options_df['QUOTE_DATE'].unique()
    
    for date in dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        # Skip if we don't have entropy or hist_vol for this date
        if date_str not in entropy_series.index or date_str not in hist_vol_series.index:
            continue
        
        # Get options data for this date
        date_options = options_df[options_df['QUOTE_DATE'] == date]
        
        # Get entropy and historical volatility for this date
        entropy = entropy_series.loc[date_str]
        hist_vol = hist_vol_series.loc[date_str]
        
        # Predict implied volatility
        implied_vol = predict_implied_volatility(iv_model, entropy, hist_vol)
        
        for _, option in date_options.iterrows():
            # Basic filters
            if option['DTE'] <= 0 or option['DTE'] > 180:  # Focus on options with reasonable DTEs
                continue
                
            S = option['UNDERLYING_LAST']
            K = option['STRIKE']
            T = option['DTE'] / 365  # Convert DTE to years
            
            # Calculate model prices using entropy-derived IV
            model_call = black_scholes_call(S, K, T, risk_free_rate, implied_vol)
            model_put = black_scholes_put(S, K, T, risk_free_rate, implied_vol)
            
            # Market prices and IVs
            market_call = option['C_MID']
            market_put = option['P_MID']
            market_call_iv = option['C_IV']
            market_put_iv = option['P_IV']
            
            # Store results
            results.append({
                'date': date_str,
                'strike': K,
                'dte': option['DTE'],
                'underlying': S,
                'entropy': entropy,
                'hist_vol': hist_vol,
                'entropy_iv': implied_vol,
                'market_call_iv': market_call_iv,
                'market_put_iv': market_put_iv,
                'model_call': model_call,
                'market_call': market_call,
                'call_error_pct': (model_call - market_call) / market_call * 100 if market_call > 0 else np.nan,
                'model_put': model_put,
                'market_put': market_put,
                'put_error_pct': (model_put - market_put) / market_put * 100 if market_put > 0 else np.nan
            })
    
    return pd.DataFrame(results)

# Function to visualize the comparison results
def visualize_comparison(comparison_df):
    """
    Visualize the comparison between model and market prices
    
    Parameters:
    - comparison_df: DataFrame with comparison results
    """
    if not ENABLE_PLOT:
        print("Plotting disabled. Set ENABLE_PLOT to True to visualize results.")
        return
    
    if comparison_df.empty:
        print("No comparison data available for visualization.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Implied Volatility Comparison
    ax1 = axes[0, 0]
    ax1.scatter(comparison_df['market_call_iv'], comparison_df['entropy_iv'], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(comparison_df['market_call_iv'].min(), comparison_df['entropy_iv'].min())
    max_val = max(comparison_df['market_call_iv'].max(), comparison_df['entropy_iv'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax1.set_title('Entropy-Derived IV vs Market Call IV')
    ax1.set_xlabel('Market Call IV')
    ax1.set_ylabel('Entropy-Derived IV')
    ax1.grid(True)
    
    # Plot 2: Call Price Comparison
    ax2 = axes[0, 1]
    ax2.scatter(comparison_df['market_call'], comparison_df['model_call'], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(comparison_df['market_call'].min(), comparison_df['model_call'].min())
    max_val = max(comparison_df['market_call'].max(), comparison_df['model_call'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax2.set_title('Model Call Price vs Market Call Price')
    ax2.set_xlabel('Market Call Price')
    ax2.set_ylabel('Model Call Price')
    ax2.grid(True)
    
    # Plot 3: Put Price Comparison
    ax3 = axes[1, 0]
    ax3.scatter(comparison_df['market_put'], comparison_df['model_put'], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(comparison_df['market_put'].min(), comparison_df['model_put'].min())
    max_val = max(comparison_df['market_put'].max(), comparison_df['model_put'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax3.set_title('Model Put Price vs Market Put Price')
    ax3.set_xlabel('Market Put Price')
    ax3.set_ylabel('Model Put Price')
    ax3.grid(True)
    
    # Plot 4: Error Distribution
    ax4 = axes[1, 1]
    ax4.hist(comparison_df['call_error_pct'].dropna(), bins=50, alpha=0.5, label='Call Error %')
    ax4.hist(comparison_df['put_error_pct'].dropna(), bins=50, alpha=0.5, label='Put Error %')
    ax4.set_title('Pricing Error Distribution (% Difference)')
    ax4.set_xlabel('Error Percentage')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualizations
    # Pricing error by moneyness (Strike/Spot)
    comparison_df['moneyness'] = comparison_df['strike'] / comparison_df['underlying']
    
    plt.figure(figsize=(12, 6))
    plt.scatter(comparison_df['moneyness'], comparison_df['call_error_pct'], alpha=0.5, label='Call Error %')
    plt.scatter(comparison_df['moneyness'], comparison_df['put_error_pct'], alpha=0.5, label='Put Error %')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Pricing Error by Moneyness')
    plt.xlabel('Moneyness (Strike/Spot)')
    plt.ylabel('Error Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Pricing error by DTE
    plt.figure(figsize=(12, 6))
    plt.scatter(comparison_df['dte'], comparison_df['call_error_pct'], alpha=0.5, label='Call Error %')
    plt.scatter(comparison_df['dte'], comparison_df['put_error_pct'], alpha=0.5, label='Put Error %')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Pricing Error by Days to Expiration')
    plt.xlabel('Days to Expiration')
    plt.ylabel('Error Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()




# Function to save comparison results to disk
def save_comparison_results(comparison_df, iv_metrics, window_size):
    """
    Save comparison results to disk
    
    Parameters:
    - comparison_df: DataFrame with comparison results
    - iv_metrics: Metrics from IV model calibration
    - window_size: Window size used for entropy calculation
    """
    # Create cache directory if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Create a cache object with results and metadata
    cache_data = {
        'comparison_results': comparison_df,
        'iv_metrics': iv_metrics,
        'window_size': window_size,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_range': f"{DATA_START_DATE} to {DATA_END_DATE}"
    }
    
    # Save to disk
    with open(COMPARISON_RESULTS_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Comparison results saved to {COMPARISON_RESULTS_FILE}")

# Function to load comparison results from disk
def load_comparison_results():
    """
    Load comparison results from disk
    
    Returns:
    - Dictionary with comparison results and metadata, or None if file not found
    """
    if not os.path.exists(COMPARISON_RESULTS_FILE):
        print(f"No cached results found at {COMPARISON_RESULTS_FILE}")
        return None
    
    try:
        with open(COMPARISON_RESULTS_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"Loaded cached results from {COMPARISON_RESULTS_FILE}")
        print(f"Generated on: {cache_data['timestamp']}")
        print(f"Data range: {cache_data['data_range']}")
        print(f"Window size: {cache_data['window_size']}")
        return cache_data
    except Exception as e:
        print(f"Error loading cached results: {str(e)}")
        return None

# Function to save calibrated IV model to disk
def save_iv_model(model, metrics):
    """
    Save calibrated IV model to disk
    
    Parameters:
    - model: Calibrated LinearRegression model
    - metrics: Model evaluation metrics
    """
    # Create cache directory if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Create a cache object with model and metadata
    cache_data = {
        'model': model,
        'metrics': metrics,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_range': f"{DATA_START_DATE} to {DATA_END_DATE}"
    }
    
    # Save to disk
    with open(MODEL_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"IV model saved to {MODEL_CACHE_FILE}")

# Function to load calibrated IV model from disk
def load_iv_model():
    """
    Load calibrated IV model from disk
    
    Returns:
    - Tuple of (model, metrics) or (None, None) if file not found
    """
    if not os.path.exists(MODEL_CACHE_FILE):
        print(f"No cached IV model found at {MODEL_CACHE_FILE}")
        return None, None
    
    try:
        with open(MODEL_CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"Loaded cached IV model from {MODEL_CACHE_FILE}")
        print(f"Generated on: {cache_data['timestamp']}")
        print(f"Data range: {cache_data['data_range']}")
        return cache_data['model'], cache_data['metrics']
    except Exception as e:
        print(f"Error loading cached IV model: {str(e)}")
        return None, None

# Modified main function with caching
def main(use_cache=True, save_results=True):
    """
    Main function to run the entire analysis
    
    Parameters:
    - use_cache: Whether to use cached results if available
    - save_results: Whether to save results to disk
    """
    try:
        # If using cache, try to load comparison results first
        if use_cache:
            cache_data = load_comparison_results()
            if cache_data is not None:
                comparison_results = cache_data['comparison_results']
                iv_metrics = cache_data['iv_metrics']
                
                # Calculate pricing error metrics
                call_mae = np.abs(comparison_results['call_error_pct']).mean()
                put_mae = np.abs(comparison_results['put_error_pct']).mean()
                
                print(f"\nPricing Error Metrics:")
                print(f"Call option MAE: {call_mae:.2f}%")
                print(f"Put option MAE: {put_mae:.2f}%")
                
                # Visualize comparison results
                visualize_comparison(comparison_results)
                
                # Example option pricing - we still need market data for this
                print("Fetching market data for example option pricing...")
                sp500, vix, spy = fetch_data()
                spy_returns = calculate_returns(spy)
                window_size = DEFAULT_WINDOW_SIZE
                spy_entropy = calculate_rolling_entropy(spy_returns, window_size=window_size)
                spy_hist_vol = realized_volatility(spy_returns, window=window_size)
                
                # Load IV model from cache or recalibrate
                iv_model, _ = load_iv_model()
                if iv_model is None:
                    print("Recalibrating IV model...")
                    sp500_returns = calculate_returns(sp500)
                    sp500_entropy = calculate_rolling_entropy(sp500_returns, window_size=window_size)
                    sp500_hist_vol = realized_volatility(sp500_returns, window=window_size)
                    vix_data = vix['Close']
                    iv_model, iv_metrics = calibrate_implied_volatility(sp500_entropy, sp500_hist_vol, vix_data)
                    if save_results:
                        save_iv_model(iv_model, iv_metrics)
                
                # Price an example option
                example_option_pricing(spy, spy_entropy, spy_hist_vol, iv_model)
                
                print("\nAnalysis complete using cached results!")
                return
        
        # If not using cache or no cache found, proceed with full calculation
        print("Fetching market data...")
        sp500, vix, spy = fetch_data()
        
        # Calculate returns
        sp500_returns = calculate_returns(sp500)
        spy_returns = calculate_returns(spy)
        
        # Calculate entropy and volatility for S&P 500
        print("Calculating entropy and volatility...")
        window_size = DEFAULT_WINDOW_SIZE
        sp500_entropy = calculate_rolling_entropy(sp500_returns, window_size=window_size)
        sp500_hist_vol = realized_volatility(sp500_returns, window=window_size)
        
        # Calculate entropy and volatility for SPY
        spy_entropy = calculate_rolling_entropy(spy_returns, window_size=window_size)
        spy_hist_vol = realized_volatility(spy_returns, window=window_size)
        
        # Process VIX data (proxy for implied volatility)
        vix_data = vix['Close']
        
        # Try to load IV model from cache
        iv_model, iv_metrics = load_iv_model() if use_cache else (None, None)
        
        if iv_model is None:
            # Calibrate implied volatility model using S&P 500 data and VIX
            print("Calibrating IV model using entropy, historical volatility, and VIX...")
            iv_model, iv_metrics = calibrate_implied_volatility(sp500_entropy, sp500_hist_vol, vix_data)
            
            # Save model if requested
            if save_results:
                save_iv_model(iv_model, iv_metrics)
        
        # Print model calibration metrics
        print("\nImplied Volatility Model Metrics:")
        print(f"R-squared: {iv_metrics['r2']:.4f}")
        print(f"RMSE: {iv_metrics['rmse']:.4f}")
        print(f"MAE: {iv_metrics['mae']:.4f}")
        print("\nModel Coefficients:")
        print(f"Intercept: {iv_metrics['coefficients']['intercept']:.6f}")
        print(f"Entropy coefficient: {iv_metrics['coefficients']['entropy']:.6f}")
        print(f"Historical volatility coefficient: {iv_metrics['coefficients']['hist_vol']:.6f}")
        
        # Process SPY options data
        print("\nProcessing SPY options data...")
        options_file_path = "spy_2020_2022.csv"  # Update path if needed
        try:
            spy_options = process_options_data(options_file_path)
            
            if not spy_options.empty:
                # Compare model prices with market prices
                print("\nComparing entropy-based option prices with market prices...")
                print("This may take a while for large datasets...")
                comparison_results = compare_option_prices(
                    spy_options, iv_model, spy_entropy, spy_hist_vol
                )
                
                # Save results if requested
                if save_results:
                    save_comparison_results(comparison_results, iv_metrics, window_size)
                
                if not comparison_results.empty:
                    # Calculate overall pricing error metrics
                    call_mae = np.abs(comparison_results['call_error_pct']).mean()
                    put_mae = np.abs(comparison_results['put_error_pct']).mean()
                    
                    print(f"\nPricing Error Metrics:")
                    print(f"Call option MAE: {call_mae:.2f}%")
                    print(f"Put option MAE: {put_mae:.2f}%")
                    
                    # Visualize comparison results
                    visualize_comparison(comparison_results)
                else:
                    print("No valid comparison results generated.")
            else:
                print("No valid options data loaded.")
                
        except FileNotFoundError:
            print(f"Options data file not found: {options_file_path}")
            print("Skipping options pricing comparison")
        
        # Example: Price a new option using the entropy-derived volatility
        example_option_pricing(spy, spy_entropy, spy_hist_vol, iv_model)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

# Helper function for example option pricing
def example_option_pricing(spy, spy_entropy, spy_hist_vol, iv_model):
    """
    Price an example option using the entropy-derived volatility
    
    Parameters:
    - spy: SPY price data
    - spy_entropy: Entropy series for SPY
    - spy_hist_vol: Historical volatility series for SPY
    - iv_model: Calibrated IV model
    """
    try:
        last_date = spy_entropy.index[-1]
        entropy_value = spy_entropy.iloc[-1]
        hist_vol_value = spy_hist_vol.loc[last_date]
        implied_vol = predict_implied_volatility(iv_model, entropy_value, hist_vol_value)
        
        current_price = spy['Close'].iloc[-1]
        strike_price = round(current_price * 1.05, 2)  # 5% OTM
        days_to_expiry = 30
        
        call_price = black_scholes_call(
            current_price, strike_price, days_to_expiry/365, RISK_FREE_RATE, implied_vol
        )
        put_price = black_scholes_put(
            current_price, strike_price, days_to_expiry/365, RISK_FREE_RATE, implied_vol
        )
        
        print(f"\nExample Option Pricing using Entropy-Derived Volatility:")
        print(f"Date: {last_date}")
        print(f"SPY Price: ${current_price:.2f}")
        print(f"Strike Price: ${strike_price:.2f}")
        print(f"Days to Expiry: {days_to_expiry}")
        print(f"Entropy: {entropy_value:.4f}")
        print(f"Historical Volatility: {hist_vol_value:.4f}")
        print(f"Entropy-Derived Implied Volatility: {implied_vol:.4f}")
        print(f"Call Option Price: ${call_price:.2f}")
        print(f"Put Option Price: ${put_price:.2f}")
    except Exception as e:
        print(f"Error pricing example option: {str(e)}")

# Usage example
if __name__ == "__main__":
    # You can control caching behavior here:
    # - use_cache=True will use cached results if available
    # - save_results=True will save new results to disk
    main(use_cache=True, save_results=True)
