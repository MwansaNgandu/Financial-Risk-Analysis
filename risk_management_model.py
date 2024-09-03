import pandas as pd
import numpy as np
import Risk_Metrics as rm
import matplotlib.pyplot as plt

# VaR analysis
hfi = rm.get_hfi_returns()

# (a) Semi-deviation
semideviation= rm.semideviation(hfi)
print("Semi-deviation of HFI returns:")
print(semideviation)

# (b) Var & CVar Analysis

# Calculate the 5th percentile of the HFI returns
hfi_percentile = np.percentile(hfi, 5, axis=0)
print("\n5th percentile of HFI returns:")
print(hfi_percentile)

# Calculate the historic VaR for the HFI returns
hfi_var_historic = rm.var_historic(hfi)
print("\nHistoric VaR for HFI returns:")
print(hfi_var_historic)

# Calculate Gaussian and Cornish-Fisher VaR for each column
hfi_var_gaussian = hfi.apply(rm.var_gaussian, level=5)
hfi_var_cornish_fisher = hfi.apply(rm.var_gaussian, level=5, modified=True)

# Create a list of the VaR calculations for each method
var_list = [hfi_var_gaussian, hfi_var_cornish_fisher, hfi_var_historic]

# Concatenate the VaR calculations into a single DataFrame
comparsion = pd.concat(var_list, axis=1)
comparsion.columns = ["Gaussian", "Cornish-Fisher", "Historic"]

# Plot the VaR calculations as a bar chart
comparsion.plot.bar(title="Hedge Fund Indices: VaR")
plt.xlabel("Hedge Fund Strategies")
plt.ylabel("Value at Risk (VaR)")
plt.tight_layout()
plt.show()