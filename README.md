# Portfolio Risk Management

Using a dataset of hedge fund indices, I had computed various risk parameters, explicitly Value at risk (VaR), drawdown and deviation from normality with Python. Using different models, I had computed non-parametric VaR, Parametric Gaussian Model VaR and Cornish-Fisher VaR, as well as plotted the VaR of all hedge fund indices.

## Risk Metrics

This file contains functions for calculating various risk metrics such as Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), semideviation, skewness, and kurtosis.

import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis as scipy_kurtosis