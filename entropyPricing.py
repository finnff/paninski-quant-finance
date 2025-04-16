from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# Data parameters
DATA_START_DATE = '2014-01-01'  # Start date for data fetching
DATA_END_DATE = '2024-12-31'    # End date for data fetching
ENABLE_PLOT = True              # Set to True to enable visualization

# Entropy estimation parameters
ENTROPY_BINS = 10               # Number of bins for discretization in entropy calculation
ENTROPY_REGULARIZATION = None   # Regularization parameter (None for automatic calculation)
MIN_SAMPLES_FOR_CORRELATION = 3 # Minimum samples required for correlation calculation

# Rolling window parameters
ROLLING_WINDOW_SIZES = [5,10,20, 30, 60]  # Window sizes for rolling entropy calculation
DEFAULT_WINDOW_SIZE = 20         # Default window size for visualization

# Lead-lag analysis parameters
MAX_LAG_DAYS = 10               # Maximum days to test for lead-lag relationships
LAG_STEP = 1                    # Step size between lag tests

# Trading strategy parameters #TODO: wip
ENTROPY_THRESHOLD = 0.7         # Threshold for trading signals (normalized entropy)


def fetch_data(start=DATA_START_DATE, end=DATA_END_DATE):
    sp500 = yf.download("^GSPC", start=start, end=end)
    vix = yf.download("^VIX", start=start, end=end)
    #converto to pd
    sp500 = pd.DataFrame(sp500)
    vix = pd.DataFrame(vix)
    return sp500, vix

#
try:
    sp500, vix = fetch_data()
    #print head of the data + dimensions
    print(f"SP500: {sp500.head()} \n {sp500.shape}") 
    print(f"VIX: {vix.head()} \n {vix.shape}")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit(1)


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

# Calculate correlation between entropy and VIX
def calculate_correlation(entropy_series, vix_series):
    """Calculate correlation metrics between entropy and VIX"""
    # Align dates first
    common_dates = entropy_series.index.intersection(vix_series.index)
    print(f"Common dates count: {len(common_dates)}")
    aligned_entropy = entropy_series.loc[common_dates].values.reshape(-1)
    aligned_vix = vix_series.loc[common_dates].values.reshape(-1)
    print(f"Aligned entropy shape: {aligned_entropy.shape}")
    print(f"Aligned vix shape: {aligned_vix.shape}")
    
    try:
        pearson_corr, pearson_p = stats.pearsonr(aligned_entropy, aligned_vix)
        spearman_corr, spearman_p = stats.spearmanr(aligned_entropy, aligned_vix)
    except ValueError:
        # Handle case where inputs are constant or have other issues
        pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
    
    return {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p
    }


# Analyze lead-lag relationship
def analyze_lead_lag(entropy_series, vix_series, max_lag=10):
    """Analyze lead-lag relationship between entropy and VIX"""
    # Ensure both series are aligned on the same index
    common_index = entropy_series.index.intersection(vix_series.index)
    entropy_aligned = entropy_series.loc[common_index]
    vix_aligned = vix_series.loc[common_index]
    
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            # VIX leads entropy - shift VIX forward
            shifted_vix = vix_aligned.copy()
            shifted_entropy = entropy_aligned.shift(abs(lag))
            # Drop NaNs from shifting and align data
            valid_idx = shifted_entropy.dropna().index
            aligned_vix = shifted_vix.loc[valid_idx].values.reshape(-1)
            aligned_entropy = shifted_entropy.loc[valid_idx].values.reshape(-1)
            if len(aligned_vix) > 1 and len(aligned_entropy) > 1:  # Need at least 2 points for correlation
                corr = np.corrcoef(aligned_vix, aligned_entropy)[0, 1]
            else:
                corr = np.nan
        elif lag > 0:
            # Entropy leads VIX - shift entropy forward
            shifted_entropy = entropy_aligned.copy()
            shifted_vix = vix_aligned.shift(lag)
            # Drop NaNs from shifting and align data
            valid_idx = shifted_vix.dropna().index
            aligned_vix = shifted_vix.loc[valid_idx].values.reshape(-1)
            aligned_entropy = shifted_entropy.loc[valid_idx].values.reshape(-1)
            if len(aligned_vix) > 1 and len(aligned_entropy) > 1:
                corr = np.corrcoef(aligned_vix, aligned_entropy)[0, 1]
            else:
                corr = np.nan
        else:
            # No lag - ensure we're working with numpy arrays
            aligned_vix = vix_aligned.values.reshape(-1)
            aligned_entropy = entropy_aligned.values.reshape(-1)
            if len(aligned_vix) > 1 and len(aligned_entropy) > 1:
                corr = np.corrcoef(aligned_vix, aligned_entropy)[0, 1]
            else:
                corr = np.nan
        
        correlations.append(corr)
    
    return pd.Series(correlations, index=lags)

#TODO: Implement a trading strategy based on entropy signals, 
#currently only buys once then hodls, should be able to sell as well

def backtest_strategy(sp500, entropy_series, threshold=ENTROPY_THRESHOLD):
    # Get dates where entropy exceeds threshold (normalized)
    norm_entropy = normalize_series(entropy_series)
    
    # Create a trading signal series (1 = in market, 0 = out of market)
    signal = pd.Series(0, index=sp500.index)
    
    # Set signals: 1 when entropy > threshold, 0 when entropy <= threshold
    common_dates = signal.index.intersection(norm_entropy.index)
    signal.loc[common_dates] = (norm_entropy.loc[common_dates] > threshold).astype(int)
    
    # Calculate strategy returns
    sp500_returns = calculate_returns(sp500)
    strategy_returns = sp500_returns * signal.shift(1).dropna()  # Apply signal with 1-day delay
    
    # Calculate metrics
    cumulative_market = (1 + sp500_returns).cumprod() - 1
    cumulative_strategy = (1 + strategy_returns).cumprod() - 1
    
    # Calculate Sharpe ratio (252 market days per year)
    sharpe_market = np.sqrt(252) * sp500_returns.mean() / sp500_returns.std()
    sharpe_strategy = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return {
        'market_returns': cumulative_market,
        'strategy_returns': cumulative_strategy,
        'sharpe_market': sharpe_market,
        'sharpe_strategy': sharpe_strategy
    }


# Main benchmark function
def benchmark_entropy_vs_vix(sp500, vix, window_sizes=[20, 40, 60]):
    """Main benchmark function comparing Paninski entropy with VIX"""
    results = {}
    
    # Calculate returns
    returns = calculate_returns(sp500)
    
    # Process VIX data
    vix_data = vix['Close']
    
    # Calculate entropy for different window sizes
    for window in window_sizes:
        print(f"\nCalculating entropy with window size {window}...")
        print(f"Returns length: {len(returns)}")
        entropy_series = calculate_rolling_entropy(returns, window_size=window)
        
        # Normalize both series
        norm_entropy = normalize_series(entropy_series)
        norm_vix = normalize_series(vix_data)
        
        # Calculate correlation
        correlation = calculate_correlation(norm_entropy, norm_vix.loc[norm_entropy.index])
        
        try:
            # Analyze lead-lag relationship
            vix_to_compare = norm_vix.loc[norm_entropy.index]
            if len(vix_to_compare) > 1:  # Need at least 2 points for correlation
                lead_lag = analyze_lead_lag(norm_entropy, vix_to_compare)
            else:
                lead_lag = pd.Series([np.nan] * 21, index=range(-10, 11))
        except Exception as e:
            print(f"Error in lead-lag analysis: {e}")
            lead_lag = pd.Series([np.nan] * 21, index=range(-10, 11))
        
        # Store results
        results[window] = {
            'entropy': entropy_series,
            'normalized_entropy': norm_entropy,
            'correlation': correlation,
            'lead_lag': lead_lag
        }
    
    return results

# Visualization function
def visualize_results(sp500, vix, results, window_size=20):
    """Visualize benchmark results"""
    if not ENABLE_PLOT:
        print("Plotting disabled. Set ENABLE_PLOT to True to visualize results.")
        return
    
    # Extract data for the specified window size
    entropy_data = results[window_size]['normalized_entropy']
    vix_data = normalize_series(vix['Close'])
    correlation = results[window_size]['correlation']
    lead_lag = results[window_size]['lead_lag']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)
    
    # Plot 1: Entropy vs VIX
    ax1 = axes[0]
    ax1.plot(entropy_data.index, entropy_data, 'b-', label='Normalized Entropy')
    ax1.plot(vix_data.index, vix_data, 'r-', label='Normalized VIX')
    ax1.set_title(f'Entropy (window={window_size}) vs VIX')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: S&P 500 with entropy overlay
    ax2 = axes[1]
    ax2.plot(sp500.index, sp500['Close'], 'g-', label='S&P 500')
    ax2.set_ylabel('S&P 500', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_title('S&P 500 with Entropy Overlay')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(entropy_data.index, entropy_data, 'b-', label='Entropy')
    ax2_twin.set_ylabel('Normalized Entropy', color='b')
    ax2_twin.tick_params(axis='y', labelcolor='b')
    ax2.grid(True)
    
    # Plot 3: Lead-Lag Analysis
    ax3 = axes[2]
    ax3.bar(lead_lag.index, lead_lag.values)
    ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax3.set_title('Lead-Lag Correlation Analysis')
    ax3.set_xlabel('Lag (days)')
    ax3.set_ylabel('Correlation')
    ax3.grid(True)
    
    # Add correlation details as text
    text = (f"Pearson Correlation: {correlation['pearson_corr']:.4f} (p={correlation['pearson_p']:.4f})\n"
            f"Spearman Correlation: {correlation['spearman_corr']:.4f} (p={correlation['spearman_p']:.4f})")
    fig.text(0.5, 0.01, text, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

# Execute the benchmark
if __name__ == "__main__":
    # Calculate returns for S&P 500
    returns = calculate_returns(sp500)
    
    # Run the benchmark
    print("Running benchmark...")
    results = benchmark_entropy_vs_vix(sp500, vix)
    
    # Find the window size with highest average correlation
    best_avg_corr = -float('inf')
    best_window = None
    window_correlations = {}
    
    # Print key findings and find best window
    for window, data in results.items():
        corr = data['correlation']
        print(f"\nWindow size {window} results:")
        print(f"Pearson correlation: {corr['pearson_corr']:.4f} (p-value: {corr['pearson_p']:.4f})")
        print(f"Spearman correlation: {corr['spearman_corr']:.4f} (p-value: {corr['spearman_p']:.4f})")
        
        # Calculate average of absolute correlation values
        # We use absolute values because negative correlation is still informative
        # TODO: maybe dont do this ?
        avg_corr = (abs(corr['pearson_corr']) + abs(corr['spearman_corr'])) / 2
        window_correlations[window] = avg_corr
        
        if avg_corr > best_avg_corr:
            best_avg_corr = avg_corr
            best_window = window
        
        # Find optimal lag
        lead_lag = data['lead_lag']
        optimal_lag = lead_lag.idxmax()
        print(f"Optimal lag: {optimal_lag} days (correlation: {lead_lag.max():.4f})")
        
        if optimal_lag < 0:
            print(f"VIX leads entropy by {abs(optimal_lag)} days")
        elif optimal_lag > 0:
            print(f"Entropy leads VIX by {optimal_lag} days")
        else:
            print("No lead-lag relationship detected")
    
    # Print best window size based on average correlation
    print(f"\nBest window size based on average correlation: {best_window} (avg correlation: {best_avg_corr:.4f})")
    print(f"Window correlations: {window_correlations}")
    
    # Visualize results with the best window size
    visualize_results(sp500, vix, results, window_size=best_window)
    
    print("\nBenchmark complete!")
