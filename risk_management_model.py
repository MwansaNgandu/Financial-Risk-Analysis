import pandas as pd
import numpy as np
import Risk_Metrics as rm
import matplotlib.pyplot as plt

# VaR analysis
hfi = rm.get_hfi_returns()

# (a) Semi-deviation
semideviation= rm.semideviation(hfi)
print("Semi-deviation of the Hedge Fund Indices Returns:")
print(semideviation)

# (b) Var & CVar Analysis

# Calculate the 5th percentile of the HFI returns
hfi_percentile = np.percentile(hfi, 5, axis=0)
print("\n5th percentile of the Hedge Fund Indices Returns:")
print(hfi_percentile)

# Calculate the Conditional VaR  for the HFI returns
hfi_cvar = hfi.apply(rm.cvar_historic, level=5)
print("\nConditional VaR of the Hedge Fund Indices Returns:")
print(hfi_cvar)

# Calculate the historic VaR for the HFI returns
hfi_var_historic = rm.var_historic(hfi)
print("\nHistoric VaR of the Hedge Fund Indices Returns:")
print(hfi_var_historic)

# Calculate Gaussian VaR for each strategy
hfi_var_gaussian = hfi.apply(rm.var_gaussian, level=5)
print("\nGaussian VaR of the Hedge Fund Indices Returns:")
print(hfi_var_gaussian)

# Calculate Cornish-Fisher VaR for each strategy
hfi_var_cornish_fisher = hfi.apply(rm.var_gaussian, level=5, modified=True)
print("\nCornish-Fisher VaR of the Hedge Fund Indices Returns:")
print(hfi_var_cornish_fisher)

# Create a list of the VaR calculations for each method
var_list = [hfi_var_gaussian, hfi_var_cornish_fisher, hfi_var_historic, hfi_cvar]

# Concatenate the VaR calculations into a single DataFrame
comparsion = pd.concat(var_list, axis=1)
comparsion.columns = ["Gaussian", "Cornish-Fisher", "Historic", "Conditional VaR"]

# Plot the VaR calculations as a bar chart
comparsion.plot.bar(figsize=(13, 5), grid=True, title="Hedge Fund Indices: VaR")
plt.xlabel("Hedge Fund Strategies")
plt.ylabel("%")
plt.tight_layout()
plt.show()