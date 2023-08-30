#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
import timesynth as ts
import pandas as pd
np.random.seed()



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



print("FINEEEEEEEEEE")


# %%

signal = ts.signals.PseudoPeriodic(amplitude=1, frequency=0.25)
noise=None
time_sampler = ts.TimeSampler(stop_time=20)
regular_time_samples = time_sampler.sample_regular_time(num_points=100)
timeseries = ts.TimeSeries(signal_generator=signal, noise_generator=noise)
samples, signals, errors = timeseries.sample(regular_time_samples)
fig = plot_time_series(regular_time_samples, samples, "")
fig.write_image("pseudo_periodic.png")
fig.show()



signal = ts.signals.AutoRegressive(ar_param=[1.5])
noise=None
time_sampler = ts.TimeSampler(stop_time=20)
regular_time_samples = time_sampler.sample_regular_time(num_points=100)
timeseries = ts.TimeSeries(signal_generator=signal, noise_generator=noise)
samples, signals, errors = timeseries.sample(regular_time_samples)
fig = plot_time_series(regular_time_samples, samples, "")
fig.write_image("auto_regressive1.png")
fig.show()



#signal = ts.signals.AutoRegressive(ar_param=[1.5, -0.75])
signal = ts.signals.AutoRegressive(ar_params=[1.5, -0.75])
noise=None
time_sampler = ts.TimeSampler(stop_time=20)
regular_time_samples = time_sampler.sample_regular_time(num_points=100)
timeseries = ts.TimeSeries(signal_generator=signal, noise_generator=noise)
samples, signals, errors = timeseries.sample(regular_time_samples)
fig = plot_time_series(regular_time_samples, samples, "")
fig.write_image("auto_regressive2.png")
fig.show()


#%%

pseudo_samples, regular_time_samples, _, _ = generate_timeseries(signal=ts.signals.PseudoPeriodic(amplitude=1, frequency=0.25), noise=ts.noise.GaussianNoise(std=0.3))

ar_samples, regular_time_samples, _, _ = generate_timeseries(signal=ts.signals.AutoRegressive(ar_param=[1.5, -0.75]))
timeseries_ = pseudo_samples*2+ar_samples

#%%

signal=ts.signals.Sinusoidal(amplitude=1, frequency=0.25)
noise=ts.noise.GaussianNoise(std=0.3)
sinusoidal_samples, regular_time_samples, _, _ = generate_timeseries(signal=signal, noise=noise)
trend = regular_time_samples*0.4
ts = sinusoidal_samples+trend

plot_time_series(regular_time_samples, ts, "Sinusoidal with Trend and White Noise")



# %%
