"""
This file contains functions for calculating various risk metrics for return series,
including Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), semideviation,
skewness, and kurtosis.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis as scipy_kurtosis

def drawdown(return_series: pd.Series, initial_wealth = 1000): 
    # Compute the drawdown of a return series
    '''Compute the drawdown of a return series.

    Parameters:
    return_series (pd.Series): The return series.
    initial_wealth (float, optional): The initial wealth value. Defaults to 1000.

    Returns:
    pd.DataFrame: A DataFrame containing the wealth, peaks, and drawdowns.
    '''
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index- previous_peaks)/ previous_peaks
    return pd.DataFrame({
        "Wealth" : wealth_index,
        "Peaks" : previous_peaks,
        "Drawdowns" : drawdowns
    })

def get_ffme_returns():
    # Load the Fama-French dataset for the returns of the 10 portfolios formed on ME
    try:
        me_m = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                header= 0, index_col = 0, na_values = -99.99)
        rets = me_m[['Lo 10', 'Hi 10']]
        rets.columns = ['SmallCap', 'LargeCap']
        rets = rets/100
        rets.index = pd.to_datetime(rets.index,format="%Y%m").to_period('M')
        return rets
    except FileNotFoundError:
        print("File not found.")
        return None

def get_hfi_returns():
    """
    Load the EDHEC Hedge Fund Index returns.

    Returns:
    pd.DataFrame: A DataFrame containing the returns.
    """
    try:
        hfi = pd.read_csv("edhec-hedgefundindices.csv")
        hfi=hfi.set_index('date')
        hfi = hfi/100
        hfi.index=(pd.to_datetime(hfi.index))
        hfi.index = hfi.index.to_period('M')
        return hfi
    except FileNotFoundError:
        print("File not found.")
        return None

def semideviation(r):
    """
    Compute the semideviation of a return series.

    Parameters:
    r (pd.Series): The return series.

    Returns:
    float: The semideviation.
    """
    is_negetive= r<0
    return np.std(r[is_negetive], ddof=1)

def skewness(r):
    """
    Compute the skewness of a return series.

    Parameters:
    r (pd.Series): The return series.

    Returns:
    float: The skewness.
    """
    return skew(r)

def kurtosis(r):
    """
    Compute the kurtosis of a return series.

    Parameters:
    r (pd.Series): The return series.

    Returns:
    float: The kurtosis.
    """
    return scipy_kurtosis(r)

def is_normal(r,level =0.01, alternative = 'two-sided'):
    """
    Apply the Jarque-Bera test to determine if a return series is normal.

    Parameters:
    r (pd.Series): The return series.
    level (float, optional): The significance level. Defaults to 0.01.
    alternative (str, optional): The alternative hypothesis. Defaults to 'two-sided'.

    Returns:
    bool: Whether the return series is normal.
    """
    statistic, p_value = norm.jarque_bera(r)
    if alternative == 'two-sided':
        return p_value > level
    elif alternative == 'greater':
        return p_value > level / 2
    elif alternative == 'less':
        return p_value < level / 2
    else:
        raise ValueError("Invalid alternative hypothesis")

def var_historic(r,level= 5):
    """
    Compute the historic Value-at-Risk of a return series.

    Parameters:
    r (pd.Series or pd.DataFrame): The return series.
    level (float, optional): The confidence level. Defaults to 5.

    Returns:
    float or pd.Series: The historic Value-at-Risk.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def var_gaussian(r, level=5, modified = False):
    """
    Compute the parametric Gaussian Value-at-Risk of a return series.

    Parameters:
    r (pd.Series): The return series.
    level (float, optional): The confidence level. Defaults to 5.
    modified (bool, optional): Whether to use the modified formula. Defaults to False.

    Returns:
    float: The parametric Gaussian Value-at-Risk.
    """
    if not isinstance(r, pd.Series):
        raise ValueError("Input must be a pandas Series")
    if r.empty:
        raise ValueError("Input Series is empty")
    if not 0 <= level <= 100:
        raise ValueError("Confidence level must be between 0 and 100")
    if r.hasnans:
        raise ValueError("Input Series contains missing values")

    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2-1)*s/6 +
             (z**3-3*z)*(k-3)/24 -
             (2*z**3-5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r,level=5):
    """
    Compute the historic Conditional Value-at-Risk of a return series.

    Parameters:
    r (pd.Series or pd.DataFrame): The return series.
    level (float, optional): The confidence level. Defaults to 5.

    Returns:
    float or pd.Series: The historic Conditional Value-at-Risk.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def summary_stats(r):
    """
    Compute summary statistics for a return series.

    Parameters:
    r (pd.Series): The return series.

    Returns:
    pd.Series: A Series containing the summary statistics.
    """
    stats = {
        "Mean": r.mean(),
        "Median": r.median(),
        "Standard Deviation": r.std(),
        "Skewness": skewness(r),
        "Kurtosis": kurtosis(r),
        "Semideviation": semideviation(r),
        "VaR (5%)": var_historic(r, level=5),
        "CVaR (5%)": cvar_historic(r, level=5)
    }
    return pd.Series(stats)
