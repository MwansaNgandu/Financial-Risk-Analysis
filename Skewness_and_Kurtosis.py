%load_ext autoreload #type:ignore
%autoreload 2 #type:ignore
import Risk_Metrics as op
import pandas as pd
import numpy as np
import scipy.stats

# Extract the various returns of the funds 

hfi = op.get_hfi_returns()
hfi.head()

# Concatenating the parameters

pd.concat([hfi.mean(),hfi.median(),hfi.mean>hfi.median()],axis="columns")

# SKEWNESS

# Comparing the skewness
