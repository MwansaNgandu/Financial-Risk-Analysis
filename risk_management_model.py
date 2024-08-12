import pandas as pd
import numpy as np
import Risk_Metrics as rm

# VaR analysis

hfi = rm.get_hfi_returns()

# (a) Semi-deviation
rm.semideviation(hfi)

# (b) Var & CVar Analysis

# Calculate the 5th percentile of the HFI returns
hfi_percentile = np.percentile(hfi, 5, axis=0)
print("5th percentile of HFI returns:")
print(hfi_percentile)

def var_historic(r, level=5):
    """
    Calculate the historic Value at Risk (VaR) for a given return series.

    Parameters:
    r (pd.DataFrame or pd.Series): The return series to calculate VaR for.
    level (int): The confidence level for the VaR calculation. Default is 5.

    Returns:
    pd.Series or float: The historic VaR for each asset in the return series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or Dateframe")

# Calculate the historic VaR for the HFI returns
hfi_var_historic = var_historic(hfi)
print("\nHistoric VaR for HFI returns:")
print(hfi_var_historic)

# Calculate the Gaussian and modified Cornish-Fisher VaR for the HFI returns
hfi_var_gaussian = rm.var_gaussian(hfi)
hfi_var_cornish_fisher = rm.var_gaussian(hfi, modified=True)

# Create a list of the VaR calculations for each method
var_list = [hfi_var_gaussian, hfi_var_cornish_fisher, hfi_var_historic]

# Concatenate the VaR calculations into a single DataFrame
comparsion = pd.concat(var_list, axis=1)
comparsion.columns = ["Gaussian", "Cornish-Fisher", "Historic"]

# Plot the VaR calculations as a bar chart
comparsion.plot.bar(title="Hedge Fund Indices: VaR")