#%%
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm.autonotebook import tqdm
import warnings
from utils.general import LogTime
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
import matplotlib.pyplot as plt
import joblib
import random
from IPython.display import display, HTML
# %load_ext autoreload
# %autoreload 2
np.random.seed(42)
tqdm.pandas()

from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


os.makedirs("../imgs/chapter_6", exist_ok=True)
preprocessed = Path("../data/london_smart_meters/preprocessed")

length = 100
index = pd.date_range(start="2021-11-03", periods=length)
# White Noise
y_random = pd.Series(np.random.randn(length), index=index)
# White Noise
y_random_2 = pd.Series(np.random.randn(length), index=index)
# White Noise+Trend
_y_random = pd.Series(np.random.randn(length), index=index)
t = np.arange(len(_y_random))
y_trend = _y_random+t*_y_random.mean()*0.8
# Heteroscedastic
_y_random = pd.Series(np.random.randn(length), index=index)
t = np.arange(len(_y_random))
y_hetero = (_y_random*t)
#WhiteNoise + Seasonal
_y_random = pd.Series(np.random.randn(length), index=index)
t = np.arange(len(_y_random))
y_seasonal = (_y_random+1.9*np.cos((2*np.pi*t)/(length/4)))
#unit root
_y_random = pd.Series(np.random.randn(length), index=index)
y_unit_root = _y_random.cumsum()


def generate_autoregressive_series(length, phi):
    x_start = random.uniform(-1, 1)
    y = []
    for i in range(length):
        t = x_start*phi+random.uniform(-1, 1)
        y.append(t)
        x_start = t
    return np.array(y)

y_08=generate_autoregressive_series(length, phi=0.8)
y_10=generate_autoregressive_series(length, phi=1.0)

y = y_10
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
result

mean = 0
std_dev = 1
num_samples = 10000

x = np.linspace(-3.5, 3.5, 1000)  # Adjust the range as needed
pdf = norm.pdf(x, mean, std_dev)
data = np.random.normal(mean, std_dev, num_samples)
hist, bin_edges = np.histogram(data, bins=30, density=True)
interp_func = interp1d(bin_edges[:-1], hist, kind='cubic')
pdf_interpolated = interp_func(x)


def normal_pdf(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

params, params_covariance = curve_fit(normal_pdf, bin_centers, hist, p0=[np.mean(data), np.std(data)])
mu_fit, sigma_fit = params
percentile_95 = norm.ppf(0.95, loc=mean, scale=std_dev)

fig, ax = plt.subplots(1, 1, figsize=(16,12))
#fig = plt.figure(figsize=(16, 12))
#ax.hist(data, label='histogram',bins=30, density=True, alpha=0.6, color='b')
ax.plot(x, pdf, label='PDF', color='r')
ax.plot(x, pdf_interpolated, label='Interpolated PDF', color='g', linestyle='--')
ax.plot(x, normal_pdf(x, mu_fit, sigma_fit), 'cyan', label='Fitted Normal')
ax.fill_between(x, pdf, where=(x >= percentile_95), alpha=0.2, color='orange', label='95th Percentile')
ax.axvline(x=percentile_95, color='orange', linestyle='--', label='95th Percentile Value')
ax.set_xlabel('X')
ax.set_ylabel('Probability Density')
ax.set_title('Normal Distribution PDF and Histogram')
ax.legend()
#plt.text(-3, 0.4, 'Interpolated PDF', color='g')
ax.grid(True)
plt.savefig('../imgs/normal_distribution_plot.png')
plt.show()



# %%
