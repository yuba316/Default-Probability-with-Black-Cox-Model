import os
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

path = r"D:\NYU\21-Spring\7801-PDE\final\pd"

# 0. read in data
BS = pd.read_csv(os.path.join(path, "XRAY_quarterly_balance-sheet.csv"), index_col=0)  # quarterly Balance Sheet data
Asset = BS.loc["TotalAssets"][:25]  # only use data after merger since 02/29/2016
Equity = BS.loc["TotalEquityGrossMinorityInterest"][:25]
D = BS["03/31/2022"]["TotalLiabilitiesNetMinorityInterest"]
C = BS["03/31/2022"]["\tCurrentLiabilities"]

quote = pd.read_csv(os.path.join(path, "XRAY.csv"))  # daily quote price from 03/31/16 to 05/12/22
quote["return"] = quote["Adj Close"].pct_change()  # daily return

rf = pd.read_csv(os.path.join(path, "^TNX.csv"))  # 10-year Treasury yield
rf = rf["Adj Close"].mean()*0.01  # mean value for this year

# 1. calculate equity volatility
# Method 1: standard deviation of quarterly equity
sigma_e_1 = 2*np.std(Equity/Equity.shift(-1)-1)
print("Volatility Method 1: ", sigma_e_1)

# Method 2: standard deviation of daily return (03/31/16 ~ 03/31/22)
sigma_e_2 = np.sqrt(252)*quote[quote["Date"]<="2022-03-31"]["return"].std()
print("Volatility Method 2: ", sigma_e_2)

# Method 3: standard deviation of daily return (03/31/22 ~ 05/12/22)
sigma_e_3 = np.sqrt(252)*quote[quote["Date"]>"2022-03-31"]["return"].std()
print("Volatility Method 3: ", sigma_e_3)

# Method 4: implied volatility of option (See XRAY_vix.xlsx for more details)
vix_june, vix_july = 0.235933178, 0.221560587
sigma_e_4 = np.sqrt((vix_june+vix_july)/2)
print("Volatility Method 4: ", sigma_e_4)

# 2. calculate volatility of company value
sigma_v = 2*np.std(Asset/Asset.shift(-1)-1)
T, E, V = 1, Equity[0], Asset[0]

# Method 1. Linear Regression
plt.figure(figsize=(12, 6))
plt.plot(Asset, Equity, ".")
plt.xlabel("Total Asset")
plt.ylabel("Total Equity")
plt.title("Asset - Equity")
plt.show()
# Use beta of E to V as dE/dV (OLS Regression E ~ c + V)
X = sm.add_constant(Asset.values)
reg = sm.OLS(Equity.values, X)
res = reg.fit()
dE_dV = res.params[1]
sigma_1 = sigma_e_3*E/V/dE_dV

# Method 2. Estimation
def eu(S,K,T,sigma,r,CorP=True):  # vanilla european option
    d1 = (np.log(S / K) + (r + 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sign = 1 if CorP else -1
    return sign*(S*norm.cdf(sign*d1)-K*np.exp(-r*T)*norm.cdf(sign*d2))

def digit(S,K,T,sigma,r,CorP=True): # Digital Option
    d1 = (np.log(S / K) + (r + 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sign = 1 if CorP else -1
    return np.exp(-r*T)*norm.cdf(sign*d2)

def eu_delta(S,K,T,sigma,r,CorP=True):  # vanilla european option delta
    d1 = (np.log(S / K) + (r + 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    sign = 1 if CorP else -1
    return norm.cdf(sign*d1)

def mdo_delta(V,K,D,T,sigma,r,gamma):  # dE/dV
    a = -(r-gamma-0.5*sigma**2)/sigma**2
    vt = K*np.exp(-gamma*T)
    return eu_delta(V,D,T,sigma,r)-2*a/vt*((V/vt)**(2*a-1))*eu((K*np.exp(-gamma*T))**2/V,D,T,sigma,r)+\
           ((V/vt)**(2*a-2))*eu_delta((K*np.exp(-gamma*T))**2/V,D,T,sigma,r)

FCF, Pay_Debt = 545000000, 299000000
Long_T = D/(FCF+Pay_Debt)
K = C+(D-C)*np.exp(-rf*(Long_T-T))
sigma_2 = sigma_e_3*E/V/mdo_delta(V,K,K,T,sigma_v,rf,rf)

# 3. theoretical value
def mdo(V,K,D,T,sigma,r,gamma):  # Equity Value: theoretical value of down-and-out call option with moving barrier
  a = -(r-gamma-0.5*sigma**2)/sigma**2
  return eu(V,D,T,sigma,r) - eu((K*np.exp(-gamma*T))**2/V,D,T,sigma,r) * (V/(K*np.exp(-gamma*T)))**(2*a)

def PD(V,K,D,T,sigma,r,gamma):  # Probability of Default: 1 - theoretical value of down-and-out digital call option with moving barrier
  a = -(r-gamma-0.5*sigma**2)/sigma**2
  return 1 - np.exp(r*T)* (digit(V,D,T,sigma,r) - digit((K*np.exp(-gamma*T))**2/V,D,T,sigma,r) * (V/(K*np.exp(-gamma*T)))**(2*a))

print("Moody’s Idealized Cumulative Expected Default Rates (1 Year)")
print("Baa2: 0.1700%  Baa3: 0.4200%")
print("Probability of Default within 1 Year: {}%".format(round(PD(V,K,K,T,sigma_1,rf,rf)*100,4)))