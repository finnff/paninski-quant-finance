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

# extracted csv from zip, see readme
SPY_OPTIONS_FILE = "spy_2020_2022.csv"  
# Data parameters
DATA_START_DATE = '2020-01-01'  # Start date for data fetching
DATA_END_DATE = '2024-12-31'    # End date for data fetching
ENABLE_PLOT = True              # Set to True to enable visualization

# Entropy estimation parameters
ENTROPY_BINS = 10               # Number of bins for discretization in entropy calculation
ENTROPY_REGULARIZATION = None   # Regularization parameter (None for automatic calculation)
MIN_SAMPLES_FOR_CORRELATION = 3 # Minimum samples required for correlation calculation

# rolling window parameters
ROLLING_WINDOW_SIZES = [5,6,7,8,9,10,11,12,13,14,15,20, 30, 60]  # Window sizes for rolling entropy calculation
DEFAULT_WINDOW_SIZE = 10         # Default window size for visualization

# Lead-lag analysis parameters
MAX_LAG_DAYS = 10               # Maximum days to test for lead-lag relationships
LAG_STEP = 1                    # Step size between lag tests


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
  # Scale VIX by dividing by 100 to convert from percentage to decimal
    vix_values = np.array(vix_series.loc[common_idx]).flatten() / 100.0
    
    print(f"Original VIX range: {float(vix_series.loc[common_idx].min()):.2f} to {float(vix_series.loc[common_idx].max()):.2f}")
    print(f"Scaled VIX range: {float(vix_values.min()):.4f} to {float(vix_values.max()):.4f}")
    
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
        return model.predict(X)[0]



    # Black-Scholes option pricing functions
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
            sample = ''.join([f.readline() for _ in range(2)])
        print(f"Sample of options file:\n{sample}")
        
        # Read CSV file with better error handling
        options_df = pd.read_csv(file_path, sep=',', low_memory=False)
        # print(f"Original columns: {options_df.columns.tolist()}")
        
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

# Updated compare_option_prices function with better error handling
def compare_option_prices(options_df, iv_model, entropy_series, hist_vol_series, risk_free_rate=RISK_FREE_RATE,
                          min_option_price=1.0, max_error_pct=500):
    """
    Compare model-derived option prices with market prices
    
    Parameters:
    - options_df: DataFrame containing SPY options data
    - iv_model: Calibrated implied volatility model
    - entropy_series: Series of entropy values
    - hist_vol_series: Series of historical volatility values
    - risk_free_rate: Risk-free interest rate
    - min_option_price: Minimum option price to consider (filters out cheap options)
    - max_error_pct: Maximum allowed percentage error (to filter outliers)
    
    Returns:
    - DataFrame with comparison results
    """
    results = []
    
    # Get unique dates in options data - use the column name we know exists
    if 'QUOTE_DATE' not in options_df.columns:
        print(f"Column 'QUOTE_DATE' not found. Available columns: {options_df.columns.tolist()}")
        return pd.DataFrame()
        
    dates = options_df['QUOTE_DATE'].unique()
    
    # Initialize counters for filtering statistics
    total_options = 0
    filtered_by_dte = 0
    filtered_by_price = 0
    filtered_by_moneyness = 0
    filtered_by_error = 0
    retained_options = 0
    
    for date in dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        # Skip if we don't have entropy or hist_vol for this date
        if date_str not in entropy_series.index or date_str not in hist_vol_series.index:
            continue
        
        # Get options data for this date
        date_options = options_df[options_df['QUOTE_DATE'] == date]
        total_options += len(date_options)
        
        # Get entropy and historical volatility for this date
        entropy = entropy_series.loc[date_str]
        hist_vol = hist_vol_series.loc[date_str]
        
        # If entropy or hist_vol is a Series, get the scalar value
        if isinstance(entropy, pd.Series):
            entropy = entropy.iloc[0]
        if isinstance(hist_vol, pd.Series):
            hist_vol = hist_vol.iloc[0]
        
        # Predict implied volatility
        implied_vol = predict_implied_volatility(iv_model, entropy, hist_vol)
        
        for _, option in date_options.iterrows():
            # Filter 1: Basic DTE filter - Focus on options with reasonable DTEs
            if option['DTE'] <= 0 or option['DTE'] > 180:
                filtered_by_dte += 1
                continue
                
            S = option['UNDERLYING_LAST']
            K = option['STRIKE']
            T = option['DTE'] / 365  # Convert DTE to years
            
            # Filter 2: In-the-money filter
            # For calls: Strike < Stock price (K < S)
            # For puts: Strike > Stock price (K > S)
            call_in_the_money = K < S
            put_in_the_money = K > S
            
            # Market prices
            market_call = option['C_MID']
            market_put = option['P_MID']
            
            # Filter 3: Price filter - Remove very cheap options
            call_price_valid = market_call >= min_option_price if not pd.isna(market_call) else False
            put_price_valid = market_put >= min_option_price if not pd.isna(market_put) else False
            
            # Apply filters
            process_call = call_in_the_money and call_price_valid
            process_put = put_in_the_money and put_price_valid
            
            if not (process_call or process_put):
                if not (call_in_the_money or put_in_the_money):
                    filtered_by_moneyness += 1
                else:
                    filtered_by_price += 1
                continue
            
            # Check for extremely high implied volatility that could break Black-Scholes
            if implied_vol > 2.0:  # Cap at 200% volatility
                implied_vol = 2.0
            
            # Calculate model prices using entropy-derived IV
            try:
                model_call = black_scholes_call(S, K, T, risk_free_rate, implied_vol) if process_call else np.nan
                model_put = black_scholes_put(S, K, T, risk_free_rate, implied_vol) if process_put else np.nan
            except Exception as e:
                # Skip options that cause calculation errors
                print(f"Error calculating price for option {K}/{S} (DTE={option['DTE']}): {str(e)}")
                continue
            
            # Market IVs
            market_call_iv = option['C_IV'] if process_call else np.nan
            market_put_iv = option['P_IV'] if process_put else np.nan
            
            # Calculate errors only for processed options
            call_error_pct = np.nan
            put_error_pct = np.nan
            
            if process_call and market_call > 0:
                call_error_pct = (model_call - market_call) / market_call * 100
                # Filter 4: Error magnitude filter
                if abs(call_error_pct) > max_error_pct:
                    filtered_by_error += 1
                    process_call = False
                    call_error_pct = np.nan
            
            if process_put and market_put > 0:
                put_error_pct = (model_put - market_put) / market_put * 100
                # Filter 4: Error magnitude filter
                if abs(put_error_pct) > max_error_pct:
                    filtered_by_error += 1
                    process_put = False
                    put_error_pct = np.nan
            
            # Only add options that passed all filters
            if process_call or process_put:
                # Store results
                results.append({
                    'date': date_str,
                    'strike': K,
                    'dte': option['DTE'],
                    'underlying': S,
                    'moneyness': K / S,
                    'entropy': entropy,
                    'hist_vol': hist_vol,
                    'entropy_iv': implied_vol,
                    'market_call_iv': market_call_iv,
                    'market_put_iv': market_put_iv,
                    'model_call': model_call if process_call else np.nan,
                    'market_call': market_call if process_call else np.nan,
                    'call_error_pct': call_error_pct,
                    'model_put': model_put if process_put else np.nan,
                    'market_put': market_put if process_put else np.nan,
                    'put_error_pct': put_error_pct,
                    'call_in_the_money': call_in_the_money,
                    'put_in_the_money': put_in_the_money
                })
                
                retained_options += 1
    
    # Print filtering statistics
    print(f"\nOption Filtering Statistics:")
    print(f"Total options considered: {total_options}")
    print(f"Filtered by DTE (<=0 or >180 days): {filtered_by_dte} ({filtered_by_dte/total_options*100:.1f}%)")
    print(f"Filtered by moneyness (out-of-the-money): {filtered_by_moneyness} ({filtered_by_moneyness/total_options*100:.1f}%)")
    print(f"Filtered by price (<${min_option_price}): {filtered_by_price} ({filtered_by_price/total_options*100:.1f}%)")
    print(f"Filtered by error magnitude (>{max_error_pct}%): {filtered_by_error}")
    print(f"Options retained for analysis: {retained_options} ({retained_options/total_options*100:.1f}%)")
    
    return pd.DataFrame(results)


# More robust visualization function that handles empty datasets
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
    try:
        plt.figure(figsize=(8, 6))  # Use smaller figure size to avoid memory issues
        
        # Prepare call and put data separately
        call_data = comparison_df.dropna(subset=['call_error_pct'])
        put_data = comparison_df.dropna(subset=['put_error_pct'])
        
        print(f"Visualizing {len(call_data)} call options and {len(put_data)} put options")
        
        # If we have too many points, sample them to avoid memory issues
        max_points = 5000
        if len(call_data) > max_points:
            call_data = call_data.sample(max_points, random_state=42)
            print(f"Sampling {max_points} call options for visualization")
            
        if len(put_data) > max_points:
            put_data = put_data.sample(max_points, random_state=42)
            print(f"Sampling {max_points} put options for visualization")
        
        # Plot histogram of call errors with reasonable bins
        if not call_data.empty:
            call_errors = call_data['call_error_pct']
            call_errors_clipped = np.clip(call_errors, -100, 100)
            plt.hist(call_errors_clipped, bins=30, alpha=0.5, label='Call Error %')
            
        # Plot histogram of put errors with reasonable bins
        if not put_data.empty:
            put_errors = put_data['put_error_pct']
            put_errors_clipped = np.clip(put_errors, -100, 100)
            plt.hist(put_errors_clipped, bins=30, alpha=0.5, label='Put Error %')
        
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Pricing Error Distribution (% Difference, Clipped to ±100%)')
        plt.xlabel('Error Percentage')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('error_distribution.png')  # Save to file instead of displaying
        print("Plot saved to error_distribution.png")
        
        # Calculate and print basic statistics
        print("\nError Statistics (In-the-Money Options):")
        
        if not call_data.empty:
            call_errors = call_data['call_error_pct']
            print(f"Call Options:")
            print(f"  Count: {len(call_errors)}")
            print(f"  Mean Error: {call_errors.mean():.2f}%")
            print(f"  Median Error: {call_errors.median():.2f}%")
            print(f"  Mean Absolute Error: {call_errors.abs().mean():.2f}%")
            print(f"  Standard Deviation: {call_errors.std():.2f}%")
            print(f"  5th Percentile: {call_errors.quantile(0.05):.2f}%")
            print(f"  95th Percentile: {call_errors.quantile(0.95):.2f}%")
        
        if not put_data.empty:
            put_errors = put_data['put_error_pct']
            print(f"\nPut Options:")
            print(f"  Count: {len(put_errors)}")
            print(f"  Mean Error: {put_errors.mean():.2f}%")
            print(f"  Median Error: {put_errors.median():.2f}%")
            print(f"  Mean Absolute Error: {put_errors.abs().mean():.2f}%")
            print(f"  Standard Deviation: {put_errors.std():.2f}%")
            print(f"  5th Percentile: {put_errors.quantile(0.05):.2f}%")
            print(f"  95th Percentile: {put_errors.quantile(0.95):.2f}%")
            
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()   
    # Calculate and print some basic statistics
    print("\nError Statistics (In-the-Money Options):")
    
    if not call_data.empty:
        call_errors = call_data['call_error_pct'].dropna()
        print(f"Call Options:")
        print(f"  Count: {len(call_errors)}")
        print(f"  Mean Error: {call_errors.mean():.2f}%")
        print(f"  Median Error: {call_errors.median():.2f}%")
        print(f"  Mean Absolute Error: {call_errors.abs().mean():.2f}%")
        print(f"  Standard Deviation: {call_errors.std():.2f}%")
        print(f"  5th Percentile: {call_errors.quantile(0.05):.2f}%")
        print(f"  95th Percentile: {call_errors.quantile(0.95):.2f}%")
    
    if not put_data.empty:
        put_errors = put_data['put_error_pct'].dropna()
        print(f"\nPut Options:")
        print(f"  Count: {len(put_errors)}")
        print(f"  Mean Error: {put_errors.mean():.2f}%")
        print(f"  Median Error: {put_errors.median():.2f}%")
        print(f"  Mean Absolute Error: {put_errors.abs().mean():.2f}%")
        print(f"  Standard Deviation: {put_errors.std():.2f}%")
        print(f"  5th Percentile: {put_errors.quantile(0.05):.2f}%")
        print(f"  95th Percentile: {put_errors.quantile(0.95):.2f}%")
    
    # Additional visualizations with clipped error values
    
    # Pricing error by moneyness
    plt.figure(figsize=(12, 6))
    
    if not call_data.empty:
        call_errors_clipped = np.clip(call_data['call_error_pct'], -100, 100)
        plt.scatter(call_data['moneyness'], call_errors_clipped, 
                    alpha=0.5, color='blue', label='Call Error %')
    
    if not put_data.empty:
        put_errors_clipped = np.clip(put_data['put_error_pct'], -100, 100)
        plt.scatter(put_data['moneyness'], put_errors_clipped, 
                    alpha=0.5, color='orange', label='Put Error %')
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Pricing Error by Moneyness (In-the-Money Options, Clipped to ±100%)')
    plt.xlabel('Moneyness (Strike/Spot)')
    plt.ylabel('Error Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Pricing error by DTE
    plt.figure(figsize=(12, 6))
    
    if not call_data.empty:
        call_errors_clipped = np.clip(call_data['call_error_pct'], -100, 100)
        plt.scatter(call_data['dte'], call_errors_clipped, 
                    alpha=0.5, color='blue', label='Call Error %')
    
    if not put_data.empty:
        put_errors_clipped = np.clip(put_data['put_error_pct'], -100, 100)
        plt.scatter(put_data['dte'], put_errors_clipped, 
                    alpha=0.5, color='orange', label='Put Error %')
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Pricing Error by Days to Expiration (In-the-Money Options, Clipped to ±100%)')
    plt.xlabel('Days to Expiration')
    plt.ylabel('Error Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Error by implied volatility
    plt.figure(figsize=(12, 6))
    
    if not call_data.empty:
        call_errors_clipped = np.clip(call_data['call_error_pct'], -100, 100)
        plt.scatter(call_data['market_call_iv'], call_errors_clipped, 
                    alpha=0.5, color='blue', label='Call Error %')
    
    if not put_data.empty:
        put_errors_clipped = np.clip(put_data['put_error_pct'], -100, 100)
        plt.scatter(put_data['market_put_iv'], put_errors_clipped, 
                    alpha=0.5, color='orange', label='Put Error %')
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Pricing Error by Market IV (In-the-Money Options, Clipped to ±100%)')
    plt.xlabel('Market Implied Volatility')
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


# Updated example_option_pricing function to handle Series objects properly
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
        # Get the last date for which we have entropy
        last_date = spy_entropy.index[-1]
        
        # Get scalar values, handling Series objects properly
        entropy_value = float(spy_entropy.iloc[-1])
        hist_vol_value = float(spy_hist_vol.loc[last_date].iloc[0] 
                              if isinstance(spy_hist_vol.loc[last_date], pd.Series) 
                              else spy_hist_vol.loc[last_date])
        
        # Get the stock price, ensuring it's a scalar
        current_price = float(spy['Close'].iloc[-1])
        
        # Predict implied volatility
        implied_vol = predict_implied_volatility(iv_model, entropy_value, hist_vol_value)
        
        # Set strike price as a percentage of current price
        strike_price = round(current_price * 1.05, 2)  # 5% OTM
        days_to_expiry = 30
        
        # Ensure implied volatility is reasonable
        if implied_vol > 1.0:  # Cap at 100%
            print(f"Warning: Capping implied volatility from {implied_vol:.4f} to 1.0")
            implied_vol = 1.0
        
        # Display volatility in both decimal and percentage format
        print(f"Entropy-Derived Implied Volatility: {implied_vol:.4f} ({implied_vol*100:.2f}%)")
        # Calculate option prices
        call_price = black_scholes_call(
            current_price, strike_price, days_to_expiry/365, RISK_FREE_RATE, implied_vol
        )
        put_price = black_scholes_put(
            current_price, strike_price, days_to_expiry/365, RISK_FREE_RATE, implied_vol
        )
        
        # Print results using scalar values
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
        import traceback
        traceback.print_exc()



def main(use_cache=True, save_results=True, min_option_price=1.0, max_error_pct=500, 
         force_recalibrate=True, optimize_window=False):
    """
    Main function to run the entire analysis
    
    Parameters:
    - use_cache: Whether to use cached results if available
    - save_results: Whether to save results to disk
    - min_option_price: Minimum option price to consider (filters out cheap options)
    - max_error_pct: Maximum allowed percentage error (to filter outliers)
    - force_recalibrate: Whether to force recalibration of the IV model
    - optimize_window: Whether to optimize the rolling window size (True) or use the default (False)
    """
    try:
        # Fetch market data (always needed)
        print("Fetching market data...")
        sp500, vix, spy = fetch_data()
        
        # Calculate returns
        sp500_returns = calculate_returns(sp500)
        spy_returns = calculate_returns(spy)
        
        # Process SPY options data for comparison
        print("\nProcessing SPY options data...")
        options_file_path = SPY_OPTIONS_FILE
        spy_options = process_options_data(options_file_path)
        
        if spy_options.empty:
            print("No valid options data loaded.")
            return
        
        if optimize_window:
            # Store results for all window sizes
            window_results = []
            
            # Loop through all window sizes
            print("\nTesting different rolling window sizes...")
            for window_size in ROLLING_WINDOW_SIZES:
                print(f"\n==== Testing Window Size: {window_size} ====")
                
                # Calculate entropy and volatility for this window size
                print(f"Calculating entropy and volatility with window size {window_size}...")
                sp500_entropy = calculate_rolling_entropy(sp500_returns, window_size=window_size)
                sp500_hist_vol = realized_volatility(sp500_returns, window=window_size)
                spy_entropy = calculate_rolling_entropy(spy_returns, window_size=window_size)
                spy_hist_vol = realized_volatility(spy_returns, window=window_size)
                
                # Process VIX data (proxy for implied volatility)
                vix_data = vix['Close']
                
                # Calibrate model for this window size
                print(f"Calibrating IV model with window size {window_size}...")
                iv_model, iv_metrics = calibrate_implied_volatility(sp500_entropy, sp500_hist_vol, vix_data)
                
                # Print model calibration metrics
                print(f"\nImplied Volatility Model Metrics (Window Size {window_size}):")
                print(f"R-squared: {iv_metrics['r2']:.4f}")
                print(f"RMSE: {iv_metrics['rmse']:.4f}")
                print(f"MAE: {iv_metrics['mae']:.4f}")
                
                # Compare model prices with market prices
                print(f"Comparing entropy-based option prices with market prices (Window Size {window_size})...")
                comparison_results = compare_option_prices(
                    spy_options, iv_model, spy_entropy, spy_hist_vol,
                    min_option_price=min_option_price,
                    max_error_pct=max_error_pct
                )
                
            if not comparison_results.empty:
                # Calculate pricing error metrics using finite values only
                call_errors = comparison_results['call_error_pct'].dropna()
                put_errors = comparison_results['put_error_pct'].dropna()
                
                call_mae = call_errors.abs().mean() if not call_errors.empty else np.nan
                put_mae = put_errors.abs().mean() if not put_errors.empty else np.nan
                
                # Average of call and put MAE as combined metric
                combined_mae = (call_mae + put_mae) / 2 if not (np.isnan(call_mae) or np.isnan(put_mae)) else np.nan
                
                print(f"\nPricing Error Metrics (Window Size {window_size}, In-the-Money Options):")
                print(f"Call option MAE: {call_mae:.2f}%")
                print(f"Put option MAE: {put_mae:.2f}%")
                print(f"Combined MAE: {combined_mae:.2f}%")
                
                # Store results for this window size
                window_results.append({
                    'window_size': window_size,
                    'call_mae': call_mae,
                    'put_mae': put_mae,
                    'combined_mae': combined_mae,
                    'r2': iv_metrics['r2'],
                    'rmse': iv_metrics['rmse'],
                    'model_mae': iv_metrics['mae'],
                    'comparison_results': comparison_results,
                    'iv_model': iv_model,
                    'iv_metrics': iv_metrics
                })
                
                # Save results for this window size if requested
                if save_results:
                    save_comparison_results(comparison_results, iv_metrics, window_size)
                    save_iv_model(iv_model, iv_metrics)
            else:
                print(f"No valid comparison results generated for window size {window_size}.")
            
            # Find the best window size based on combined MAE
            if window_results:
                print("\n==== Summary of Results for Different Window Sizes ====")
                print("\nWindow Size | Call MAE | Put MAE | Combined MAE | R-squared")
                print("------------|----------|---------|--------------|----------")
                
                for result in window_results:
                    print(f"{result['window_size']:11} | {result['call_mae']:7.2f}% | {result['put_mae']:6.2f}% | {result['combined_mae']:11.2f}% | {result['r2']:9.4f}")
                
                # Find the window size with the lowest combined MAE
                valid_results = [r for r in window_results if not np.isnan(r['combined_mae'])]
                
                if valid_results:
                    best_result = min(valid_results, key=lambda x: x['combined_mae'])
                    best_window_size = best_result['window_size']
                    
                    print(f"\n==== Best Window Size: {best_window_size} ====")
                    print(f"Call option MAE: {best_result['call_mae']:.2f}%")
                    print(f"Put option MAE: {best_result['put_mae']:.2f}%")
                    print(f"Combined MAE: {best_result['combined_mae']:.2f}%")
                    print(f"R-squared: {best_result['r2']:.4f}")
                    
                    # Use the best window size for subsequent analysis
                    window_size = best_window_size
                    iv_model = best_result['iv_model']
                    iv_metrics = best_result['iv_metrics']
                    comparison_results = best_result['comparison_results']
                else:
                    print("\nNo valid results available to determine the best window size.")
                    # Fallback to default window size
                    window_size = DEFAULT_WINDOW_SIZE
            else:
                print("\nNo results available for any window size.")
                # Fallback to default window size
                window_size = DEFAULT_WINDOW_SIZE
        else:
            # If not optimizing, use the default window size
            window_size = DEFAULT_WINDOW_SIZE
            
            # Calculate entropy and volatility with the default window size
            print(f"Calculating entropy and volatility with default window size {window_size}...")
            sp500_entropy = calculate_rolling_entropy(sp500_returns, window_size=window_size)
            sp500_hist_vol = realized_volatility(sp500_returns, window=window_size)
            spy_entropy = calculate_rolling_entropy(spy_returns, window_size=window_size)
            spy_hist_vol = realized_volatility(spy_returns, window=window_size)
            
            # Process VIX data (proxy for implied volatility)
            vix_data = vix['Close']
            
            # Try to load IV model from cache if not optimizing
            if not force_recalibrate and use_cache:
                iv_model, iv_metrics = load_iv_model()
            else:
                iv_model = None
                
            # Calibrate model if necessary
            if iv_model is None:
                print(f"Calibrating IV model with default window size {window_size}...")
                iv_model, iv_metrics = calibrate_implied_volatility(sp500_entropy, sp500_hist_vol, vix_data)
                
                # Save model if requested
                if save_results:
                    save_iv_model(iv_model, iv_metrics)
            
            # Print model calibration metrics
            print("\nImplied Volatility Model Metrics:")
            print(f"R-squared: {iv_metrics['r2']:.4f}")
            print(f"RMSE: {iv_metrics['rmse']:.4f}")
            print(f"MAE: {iv_metrics['mae']:.4f}")
            
            # Compare model prices with market prices
            print("\nComparing entropy-based option prices with market prices...")
            comparison_results = compare_option_prices(
                spy_options, iv_model, spy_entropy, spy_hist_vol,
                min_option_price=min_option_price,
                max_error_pct=max_error_pct
            )
            
            # Save results if requested
            if save_results:
                save_comparison_results(comparison_results, iv_metrics, window_size)
        
        # Visualize the results
        if not comparison_results.empty:
            print("\nVisualizing results...")
            visualize_comparison(comparison_results)
            
            # Calculate pricing error metrics using finite values only
            call_errors = comparison_results['call_error_pct'].dropna()
            put_errors = comparison_results['put_error_pct'].dropna()
            
            call_mae = call_errors.abs().mean() if not call_errors.empty else np.nan
            put_mae = put_errors.abs().mean() if not put_errors.empty else np.nan
            
            print(f"\nFinal Pricing Error Metrics (Window Size {window_size}, In-the-Money Options):")
            print(f"Call option MAE: {call_mae:.2f}%")
            print(f"Put option MAE: {put_mae:.2f}%")
            
            # Example: Price a new option using the entropy-derived volatility
            print("\nPricing example option...")
            spy_entropy_final = calculate_rolling_entropy(spy_returns, window_size=window_size)
            spy_hist_vol_final = realized_volatility(spy_returns, window=window_size)
            example_option_pricing(spy, spy_entropy_final, spy_hist_vol_final, iv_model)
        else:
            print("\nNo valid comparison results generated for final analysis.")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configuration:
    # - force_recalibrate=True to ensure models are calibrated for each window size
    # - use_cache=False to force recalculation of results
    # - save_results=True to save the results to disk
    # - optimize_window=True to find the best window size
    main(use_cache=False, save_results=True, min_option_price=1.0, max_error_pct=500, 
         force_recalibrate=True, optimize_window=False)

