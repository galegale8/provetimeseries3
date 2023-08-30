
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None
import timesynth as ts
import pandas as pd
np.random.seed()
from pathlib import Path
from tqdm.autonotebook import tqdm
from itertools import cycle


def plot_time_series(time, values, label, legends=None):
    if legends is not None:
        assert len(legends)==len(values)
    if isinstance(values, list):
        series_dict = {"Time": time}
        for v, l in zip(values, legends):
            series_dict[l] = v
        plot_df = pd.DataFrame(series_dict)
        plot_df = pd.melt(plot_df,id_vars="Time",var_name="ts", value_name="Value")
    else:
        series_dict = {"Time": time, "Value": values, "ts":""}
        plot_df = pd.DataFrame(series_dict)
    
    if isinstance(values, list):
        fig = px.line(plot_df, x="Time", y="Value", line_dash="ts")
        fig.add_scatter(x=plot_df['Time'],y=plot_df['Value'],mode='markers')
    else:
        fig = px.line(plot_df, x="Time", y="Value")
        fig.add_scatter(x=plot_df['Time'],y=plot_df['Value'],mode='markers')
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        title={
        'text': label,
#         'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        titlefont={
            "size": 25
        },
        yaxis=dict(
            title_text="Value",
            titlefont=dict(size=12),
        ),
        xaxis=dict(
            title_text="Time",
            titlefont=dict(size=12),
        )
    )
    return fig

def generate_timeseries(signal, noise=None):
    time_sampler = ts.TimeSampler(stop_time=20)
    regular_time_samples = time_sampler.sample_regular_time(num_points=100)
    timeseries = ts.TimeSeries(signal_generator=signal, noise_generator=noise)
    samples, signals, errors = timeseries.sample(regular_time_samples)
    return samples, regular_time_samples, signals, errors

def format_plot(fig, legends, font_size=15, title_font_size=20):
    names = cycle(legends)
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": title_font_size},
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text="Value",
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            title_text="Day",
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
    )
    return fig



print("FINEEEEEEEEEE")

#%%


pseudo_samples, regular_time_samples, _, _ = generate_timeseries(signal=ts.signals.PseudoPeriodic(amplitude=1, frequency=0.25), noise=ts.noise.GaussianNoise(std=0.3))
# Generating an Autoregressive Signal
ar_samples, regular_time_samples, _, _ = generate_timeseries(signal=ts.signals.AutoRegressive(ar_param=[1.5, -0.75]))
# Combining the two signals using a mathematical equation
timeseries_ = pseudo_samples*2+ar_samples


#%%
# The code `df =
# pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx",
# skiprows=1)` is reading an Excel file from the given URL and storing it in a pandas DataFrame called
# `df`. The `skiprows=1` parameter is used to skip the first row of the Excel file, which is typically
# used for column headers.
#df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx", skiprows=1)
df2 = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx", skiprows=1, 
                    parse_dates=False)

df2.info()
df2.set_index("date", inplace=True)
df2["2010-01-04":]

# %%

df3 = pd.read_csv("https://www.data.act.gov.au/resource/94a5-zqnn.csv", sep=",", parse_dates=False)
df3 = df3.loc[df3.name=="Monash", ['datetime', 'pm2_5_1_hr']]
df3.datetime = pd.to_datetime(df3.datetime, yearfirst=True)
df3.sort_values("datetime", inplace=True)
df3.set_index("datetime", inplace=True)
df3 = df3["2023-08-17": "2023-08-19"]
#df3.reset_index()

fig = px.line(df3, x=df3.index, y="pm2_5_1_hr", title="Missing Values in PM2.5")
#fig.add_scatter(x=df3.index, y=df3['pm2_5_1_hr'].ffill(), mode='markers')
fig.add_scatter(x=df3.index, y=df3['pm2_5_1_hr'].interpolate(method="spline", order=2), mode='markers')
fig.add_scatter(x=df3.index, y=df3['pm2_5_1_hr'].interpolate(method="polynomial", order=5), mode='markers')

fig = format_plot(fig, ["Original"])

fig.write_image("missing_values.png")
fig.show()




plot_time_series(regular_time_samples, ts, "Sinusoidal with Trend and White Noise")

