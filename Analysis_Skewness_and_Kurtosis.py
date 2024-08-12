import pandas as pd
import Risk_Metrics as rm
import scipy.stats
import numpy as np

def load_data():
    """Load HFI returns and Fama-French model returns"""
    hfi_returns = rm.get_hfi_returns()
    ffme_returns = rm.get_ffme_returns()
    return hfi_returns, ffme_returns

def calculate_summary_stats(returns):
    """Calculate mean, median, and boolean column indicating if mean > median"""
    mean = returns.mean()
    median = returns.median()
    mean_gt_median = mean > median
    return pd.concat([mean, median, mean_gt_median], axis=1)

def calculate_skewness_kurtosis(returns):
    """Calculate skewness and kurtosis of returns"""
    skewness = rm.skewness(returns).sort_values()
    kurtosis = rm.kurtosis(returns).sort_values()
    return skewness, kurtosis

def check_normality(returns):
    """Check if returns follow a normal distribution"""
    return returns.aggregate(rm.is_normal)

def main():
    try:
        hfi_returns, ffme_returns = load_data()
        
        print("HFI Returns Summary:")
        summary_stats = calculate_summary_stats(hfi_returns)
        print(summary_stats)
        
        print("\nHFI Returns Skewness and Kurtosis:")
        skewness, kurtosis = calculate_skewness_kurtosis(hfi_returns)
        print("Skewness:")
        print(skewness)
        print("\nKurtosis:")
        print(kurtosis)
        
        print("\nHFI Returns Normality Check:")
        print(check_normality(hfi_returns))
        
        print("\nFama-French Model Returns Skewness:")
        skewness = rm.skewness(ffme_returns)
        print(skewness)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()