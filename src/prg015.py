
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


def format_plot(fig, legends = None, xlabel="Time", ylabel="Value", font_size=15, title_font_size=20):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_layout(
            autosize=False,
            width=900,
            height=500,
            title={
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            titlefont={
                "size": 20
            },
            legend_title = None,
            legend=dict(
                font=dict(size=font_size),
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
            ),
            yaxis=dict(
                title_text=ylabel,
                titlefont=dict(size=font_size),
                tickfont=dict(size=font_size),
            ),
            xaxis=dict(
                title_text=xlabel,
                titlefont=dict(size=font_size),
                tickfont=dict(size=font_size),
            )
        )
    return fig


# tesla_revenue = pd.read_html("https://en.wikipedia.org/wiki/Tesla,_Inc.")[4][['Year', "Revenue(mil. USD)"]]
# tesla_revenue["Year"] = tesla_revenue["Year"].str.split("[", expand=True).iloc[:,0].astype(int)
# tesla_revenue = tesla_revenue[tesla_revenue.Year>=2010]
# fig = px.line(tesla_revenue, x="Year", y="Revenue(mil. USD)", title="Tesla's Revenue in M USDs")
# fig = format_plot(fig, xlabel="Year")
# fig.write_image("imgs/chapter_3/tesla_revenue.png")
# fig.show(renderer="svg")


length = 2000
index = pd.date_range(start="2010-01-01", periods=length)
#WhiteNoise + Seasonal
_y_random = pd.Series(np.random.randn(length), index=index)
t = np.arange(len(_y_random))
y_seasonal = (_y_random+1.9*np.cos((2*np.pi*t)/(length/4)))
y_seasonal.name = 'y'
y_seasonal.info()

y=y_seasonal
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)

fig, ax = plt.subplots(1, 1, figsize=(16,12))

plt.plot(y.index, y.values,  linestyle='-')

y.values
print("FINEEEEEEEEEEEEEEE")

# %%
