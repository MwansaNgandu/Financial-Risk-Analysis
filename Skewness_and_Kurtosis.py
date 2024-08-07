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

op.skewness(hfi).sort_values() # If it was normal, then skewness of 0 would be obtain.

#Alternatively, you can use an exisiting library rather then the function in the OurModule.
scipy.stats.skew(hfi)

#To find the sample size:
hfi.shape

#Conduct another santiy check by extracting the normal distribution from numpy. The distribution would have a mean of 0, SD of 0.15 and sample size 263.
normal_rets = np.random.normal(0, .15,size=(263,1))
op.skewness(normal_rets)


op.kurtosis(hfi).sort_values()

#Check out kurtosis for normal returns
op.kurtosis(normal_rets)


#Alternatively, you can use an exisiting library rather then the function in the OurModule.
scipy.stats.kurtosis(normal_rets)

# Running a Jarque Bera Test 
'''
Jarque–Bera test is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution.
'''
scipy.stats.jarque_bera(normal_rets)
scipy.stats.jarque_bera(hfi)

'''
Are the returns of funds in the portfolio dataset normally distributed?
'''
#Test for normal distribution
hfi.aggregate(op.is_normal)

'''
Are the returns of small & large cap stocks (Fama-French Model) in the portfolio dataset normally distributed?Are the returns of small & large cap stocks (Fama-French Model) in the portfolio dataset normally distributed?
'''
#Test for normal distribution
ffme = op.get_ffme_returns()
op.skewness(ffme)