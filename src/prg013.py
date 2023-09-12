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

plot_df = pd.DataFrame({"Time":np.arange(100),"Timeseries 1": y_random, "Timeseries 2": y_trend, "Timeseries 3": y_hetero, "Timeseries 4": y_unit_root, "Timeseries 5": y_seasonal, "Timeseries 6": y_random_2})


fig = px.line(pd.melt(plot_df, id_vars="Time", value_name="Observed"), x="Time", y="Observed", facet_col="variable", facet_col_wrap=3)
fig.update_yaxes(matches=None)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=16)))
fig.update_layout(
            autosize=False,
            width=1600,
            height=800,
            yaxis=dict(
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            ),
            xaxis=dict(
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            )
        )
fig.update_annotations(font_size=20)
fig.write_image("../imgs/chapter_6/stationary_ts.png")
fig.show()


from statsmodels.tsa.stattools import adfuller
result = adfuller(y_unit_root)


# y_random y_random_2 y_trend y_hetero y_seasonal y_unit_root
from transforms.stationary_utils import check_unit_root



from transforms.target_transformations import AdditiveDifferencingTransformer, MultiplicativeDifferencingTransformer

diff_transformer = AdditiveDifferencingTransformer()
# [1:] because differencing reduces the lenght of the time series by one
y_diff = diff_transformer.fit_transform(y_unit_root)[1:]
y_ric = diff_transformer.inverse_transform(y_diff)

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


res = check_unit_root(y_08, confidence=0.05)
print(f"Stationary: {res.stationary} | p-value: {res.results[1]}")

from transforms.stationary_utils import check_trend, check_deterministic_trend

res = check_deterministic_trend(y_trend)
print(f"Stationary: {res.adf_res.stationary} | Deterministic Trend: {res.deterministic_trend}")


# y_random y_random_2 y_trend y_hetero y_seasonal y_unit_root y_08 y_10

import scipy.stats as stats


tau, p_value = stats.kendalltau(y, np.arange(len(y)))

y = y_trend
check_trend(y, confidence=0.05)
check_trend(y, confidence=0.05, mann_kendall=True)


from transforms.target_transformations import DetrendingTransformer
detrending_transformer = DetrendingTransformer(degree=1)
y_detrended = detrending_transformer.fit_transform(y, freq="1D")

check_trend(y_detrended, confidence=0.05, mann_kendall=True)
y2 = detrending_transformer.inverse_transform (y_detrended)

y_01=generate_autoregressive_series(length, phi=0.1)
y = y_01

from statsmodels.tsa.stattools import acf
r = acf(y, nlags=60, fft=False)
r = r[1:]
plot_df = pd.DataFrame(dict(x=np.arange(len(r))+1, y=r)) #
plot_df['seasonal_lag'] = False
plot_df.loc[plot_df["x"].isin([25,50]), "seasonal_lag"] = True

fig = px.bar(plot_df, x="x", y="y", pattern_shape="seasonal_lag", color="seasonal_lag", title="Auto-Correlation Plot")
fig.add_annotation(x=25, y=r[24], text="Lag 25")
fig.add_annotation(x=50, y=r[49], text="Lag 50")
fig.update_layout(
            showlegend = False,
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
            yaxis=dict(
                title_text="Auto Correlation",
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            ),
            xaxis=dict(
                title_text="Lags",
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            )
        )
fig.update_annotations(font_size=15)
fig.write_image("../imgs/chapter_6/acf_plot.png")
fig.show()



y_01=generate_autoregressive_series(length, phi=0.1)
y = y_01

from statsmodels.tsa.stattools import acf
r = acf(y, nlags=60, fft=False)
r = r[1:]
plot_df = pd.DataFrame(dict(x=np.arange(len(r))+1, y=r)) #
plot_df['seasonal_lag'] = False
fig = px.bar(plot_df, x="x", y="y", pattern_shape="seasonal_lag", color="seasonal_lag", title="Auto-Correlation Plot")
fig.show()















y_seasonal.plot()
plt.show()
kendall_tau_res = check_trend(y_seasonal, confidence=0.05)
mann_kendall_res = check_trend(y_seasonal, confidence=0.05, mann_kendall=True)
mann_kendall_seas_res = check_trend(y_seasonal, confidence=0.05, mann_kendall=True, seasonal_period=25)
print(f"Kendalls Tau: Trend: {kendall_tau_res.trend} | Direction: {kendall_tau_res.direction} | Deterministic: {kendall_tau_res.deterministic}")
print(f"Mann-Kendalls: Trend: {mann_kendall_res.trend} | Direction: {mann_kendall_res.direction} | Deterministic: {mann_kendall_res.deterministic}")
print(f"Mann-Kendalls Seasonal: Trend: {mann_kendall_seas_res.trend} | Direction: {mann_kendall_seas_res.direction} | Deterministic: {mann_kendall_seas_res.deterministic}")



import plotly.graph_objects as go
from plotly.subplots import make_subplots


fig = make_subplots(
    rows=3, cols=1, subplot_titles=("$\phi=0.8$", "$\phi=1.0$", "$\phi=1.05$")
)

fig.append_trace(
    go.Scatter(
        x=np.arange(length),
        y=generate_autoregressive_series(length, phi=0.8),
    ),
    row=1,
    col=1,
)

fig.append_trace(
    go.Scatter(
        x=np.arange(length),
        y=generate_autoregressive_series(length, phi=1.0),
    ),
    row=2,
    col=1,
)

fig.append_trace(
    go.Scatter(x=np.arange(length), y=generate_autoregressive_series(length, phi=1.05)),
    row=3,
    col=1,
)


fig.update_layout(
    height=700,
    width=700,
    showlegend=False,
    yaxis=dict(
        titlefont=dict(size=15),
        tickfont=dict(size=15),
    ),
    xaxis=dict(
        titlefont=dict(size=15),
        tickfont=dict(size=15),
    ),
)
fig.write_image("../imgs/chapter_6/ar_series_phi.png")
fig.show()



from transforms.stationary_utils import check_unit_root

res = check_unit_root(y_unit_root, confidence=0.05)

print(f"Stationary: {res.stationary} | p-value: {res.results[1]}")


from transforms.target_transformations import AdditiveDifferencingTransformer, MultiplicativeDifferencingTransformer


diff_transformer = AdditiveDifferencingTransformer()
# [1:] because differencing reduces the lenght of the time series by one
y_diff = diff_transformer.fit_transform(y_unit_root)[1:]
y_ric = diff_transformer.inverse_transform(y_diff)

fig, axs = plt.subplots(2)
y_unit_root.plot(title="Unit Root",ax=axs[0])
y_diff.plot(title="Additive Difference",ax=axs[1])
plt.tight_layout()
plt.show()
check_unit_root(y_diff)

res = check_unit_root(y_diff, confidence=0.05)


from transforms.stationary_utils import check_trend, check_deterministic_trend

ar_series = generate_autoregressive_series(length, phi=1.05)

res = check_deterministic_trend(ar_series, confidence=0.05)





# %%
