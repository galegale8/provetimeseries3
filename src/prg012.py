#%%

import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
import statsmodels.api as sm
import warnings
import random
from IPython.display import display, HTML
np.random.seed(42)
random.seed(42)
tqdm.pandas()

try:
    lclid_acorn_map = pd.read_pickle("../data/london_smart_meters/preprocessed/london_smart_meters_lclid_acorn_map.pkl")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run 02 - Preprocessing London Smart Meter Dataset.ipynb in Chapter02
    </div>
    """))

affluent_households = lclid_acorn_map.loc[lclid_acorn_map.Acorn_grouped=="Affluent", ["LCLid",'file']]
adversity_households = lclid_acorn_map.loc[lclid_acorn_map.Acorn_grouped=="Adversity", ["LCLid",'file']]
comfortable_households = lclid_acorn_map.loc[lclid_acorn_map.Acorn_grouped=="Comfortable", ["LCLid",'file']]

selected_households = pd.concat(
    [
        affluent_households.sample(50, random_state=76),
        comfortable_households.sample(50, random_state=76),
        adversity_households.sample(50, random_state=76),
    ]
)
selected_households['block']=selected_households.file.str.split("_", expand=True).iloc[:,1].astype(int)



path_blocks = [
    (p, *list(map(int, p.name.split("_")[5].split(".")[0].split("-"))))
    for p in Path("../data/london_smart_meters/preprocessed").glob(
        "london_smart_meters_merged_block*"
    )
]


household_df_l = []
for path, start_b, end_b in tqdm(path_blocks):
    block_df = pd.read_parquet(path)
    selected_households['block'].between
    mask = selected_households['block'].between(start_b, end_b)
    lclids = selected_households.loc[mask, "LCLid"]
    household_df_l.append(block_df.loc[block_df.LCLid.isin(lclids)])


block_df = pd.concat(household_df_l)
del household_df_l
block_df.head()

from utils.data_utils import compact_to_expanded

exp_block_df = compact_to_expanded(block_df, timeseries_col = 'energy_consumption',
static_cols = ["frequency", "series_length", "stdorToU", "Acorn", "Acorn_grouped", "file"],
time_varying_cols = ['holidays', 'visibility', 'windBearing', 'temperature', 'dewPoint',
       'pressure', 'apparentTemperature', 'windSpeed', 'precipType', 'icon',
       'humidity', 'summary'],
ts_identifier = "LCLid")

exp_block_df.head()

test_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==2)
val_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==1)

train = exp_block_df[~(val_mask|test_mask)]
val = exp_block_df[val_mask]
test = exp_block_df[test_mask]
print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
print(f"Max Date in Train: {train.timestamp.max()} | Min Date in Validation: {val.timestamp.min()} | Min Date in Test: {test.timestamp.min()}")

train.to_parquet("../data/london_smart_meters/preprocessed/selected_blocks_train.parquet")
val.to_parquet("../data/london_smart_meters/preprocessed/selected_blocks_val.parquet")
test.to_parquet("../data/london_smart_meters/preprocessed/selected_blocks_test.parquet")

from imputation.interpolation import SeasonalInterpolation

exp_block_df.energy_consumption.info()

block_df.energy_consumption = block_df.energy_consumption.progress_apply(lambda x: SeasonalInterpolation(seasonal_period=48*7).fit_transform(x.reshape(-1,1)).squeeze())

exp_block_df = compact_to_expanded(block_df, timeseries_col = 'energy_consumption',
static_cols = ["frequency", "series_length", "stdorToU", "Acorn", "Acorn_grouped", "file"],
time_varying_cols = ['holidays', 'visibility', 'windBearing', 'temperature', 'dewPoint',
       'pressure', 'apparentTemperature', 'windSpeed', 'precipType', 'icon',
       'humidity', 'summary'],
ts_identifier = "LCLid")

from utils.data_utils import reduce_memory_footprint

exp_block_df = reduce_memory_footprint(exp_block_df)

test_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==2)
val_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==1)

train = exp_block_df[~(val_mask|test_mask)]
val = exp_block_df[val_mask]
test = exp_block_df[test_mask]
print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
print(f"Max Date in Train: {train.timestamp.max()} | Min Date in Validation: {val.timestamp.min()} | Min Date in Test: {test.timestamp.min()}")

train.to_parquet("../data/london_smart_meters/preprocessed/selected_blocks_train_missing_imputed.parquet")
val.to_parquet("../data/london_smart_meters/preprocessed/selected_blocks_val_missing_imputed.parquet")
test.to_parquet("../data/london_smart_meters/preprocessed/selected_blocks_test_missing_imputed.parquet")

print("FINEEEEEEEEEEEEEEEEEEEEEEEEEEE  ")

#%%

exp_block_df.LCLid.unique()
from utils.data_utils import reduce_memory_footprint
exp_block_df.info(memory_usage="deep", verbose=False)
exp_block_df = reduce_memory_footprint(exp_block_df)
exp_block_df.info()
lclid_acorn_map.iloc[0]

lclid_acorn_map.LCLid.unique()
exp_block_df.iloc[0].timestamp.year

exp_block_df.timestamp.dt.year.unique()
# # Train Test Valildation Split

# We are going to keep 2014 data as the validation and test period. We have 2 months(Jan and Feb) of data in 2014. Jan is Validation and Feb is Test





####################################################

#%%
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from utils.general import LogTime
from tqdm.autonotebook import tqdm
from IPython.display import display, HTML

# %load_ext autoreload
# %autoreload 2
np.random.seed(42)
tqdm.pandas()

os.makedirs("../imgs/chapter_6", exist_ok=True)
preprocessed = Path("../data/london_smart_meters/preprocessed")



try:
    train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed.parquet")
    val_df = pd.read_parquet(preprocessed/"selected_blocks_val_missing_imputed.parquet")
    test_df = pd.read_parquet(preprocessed/"selected_blocks_test_missing_imputed.parquet")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run 01-Setting up Experiment Harness.ipynb in Chapter04
    </div>
    """))


train_df["type"] = "train"
val_df["type"] = "val"
test_df["type"] = "test"
full_df = pd.concat([train_df, val_df, test_df]).sort_values(["LCLid", "timestamp"])
del train_df, test_df, val_df

from feature_engineering.autoregressive_features import add_lags

lags = (
    (np.arange(5) + 1).tolist()
    + (np.arange(5) + 46).tolist()
    + (np.arange(5) + (48 * 7) - 2).tolist()
)





with LogTime():
    full_df, added_features = add_lags(
        full_df, lags=lags, column="energy_consumption", ts_id="LCLid", use_32_bit=True
    )
print(f"Features Created: {','.join(added_features)}")


from feature_engineering.autoregressive_features import add_rolling_features


with LogTime():
    full_df, added_features = add_rolling_features(
        full_df,
        rolls=[3, 6, 12, 48],
        column="energy_consumption",
        agg_funcs=["mean", "std"],
        ts_id="LCLid",
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")

from feature_engineering.autoregressive_features import (
    add_seasonal_rolling_features,
)


with LogTime():
    full_df, added_features = add_seasonal_rolling_features(
        full_df,
        rolls=[3],
        seasonal_periods=[48, 48 * 7],
        column="energy_consumption",
        agg_funcs=["mean", "std"],
        ts_id="LCLid",
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")



t = np.arange(25).tolist()
plot_df = pd.DataFrame({"Timesteps behind t": t})
for alpha in [0.3, 0.5, 0.8]:
    weights = [alpha * math.pow((1 - alpha), i) for i in t]
    span = (2 - alpha) / alpha
    halflife = math.log(1 - alpha) / math.log(0.5)
    plot_df[f"Alpha={alpha} | Span={span:.2f}"] = weights

fig = px.line(
    pd.melt(plot_df, id_vars="Timesteps behind t", var_name="Parameters"),
    x="Timesteps behind t",
    y="value",
    facet_col="Parameters",
)
fig.update_layout(
    autosize=False,
    width=1200,
    height=500,
    yaxis=dict(
        title_text="Weights",
        titlefont=dict(size=15),
        tickfont=dict(size=15),
    ),
    xaxis=dict(
        titlefont=dict(size=15),
        tickfont=dict(size=15),
    ),
)
fig.update_annotations(font=dict(size=16))
fig.write_image(f"../imgs/chapter_6/ewma_weights.png")
fig.show()


from feature_engineering.autoregressive_features import add_ewma


with LogTime():
    # full_df, added_features = add_ewma(full_df, alphas=[0.2, 0.5, 0.9], column="energy_consumption", ts_id="LCLid", use_32_bit=True)
    full_df, added_features = add_ewma(
        full_df,
        spans=[48 * 60, 48 * 7, 48],
        column="energy_consumption",
        ts_id="LCLid",
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")

print("FINEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")























print("FINEEEEEEEEEEEEEEEEEEEE")
###########################################

full_df.info()












#%%


import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.io as pio
pio.templates.default = "plotly_white"
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from IPython.display import display, HTML
# %load_ext autoreload
# %autoreload 2
np.random.seed()
tqdm.pandas()
print("FINEEEEEEEEEEEEE")



os.makedirs("../imgs/chapter_3", exist_ok=True)
preprocessed = Path("../data/london_smart_meters/preprocessed")
assert preprocessed.is_dir(), "You have to run 02 - Preprocessing London Smart Meter Dataset.ipynb in Chapter02 before running this notebook"

from utils import plotting_utils
from utils.data_utils import compact_to_expanded
def format_plot(fig, legends = None, xlabel="Time", ylabel="Value", figsize=(500,900), font_size=15, title_font_size=20):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_layout(
            autosize=False,
            width=figsize[1],
            height=figsize[0],
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



try:
    block_df = pd.read_parquet(preprocessed/"london_smart_meters_merged_block_0-7.parquet")
    display(block_df.head())
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run 02 - Preprocessing London Smart Meter Dataset.ipynb in Chapter02
    </div>
    """))


exp_block_df = compact_to_expanded(block_df[block_df.file=="block_7"], timeseries_col = 'energy_consumption',
static_cols = ["frequency", "series_length", "stdorToU", "Acorn", "Acorn_grouped", "file"],
time_varying_cols = ['holidays', 'visibility', 'windBearing', 'temperature', 'dewPoint',
       'pressure', 'apparentTemperature', 'windSpeed', 'precipType', 'icon',
       'humidity', 'summary'],
ts_identifier = "LCLid")

ts_df = exp_block_df[exp_block_df.LCLid=="MAC000193"].set_index("timestamp")

ts_df["year"] = ts_df.index.year
ts_df["month_name"] = ts_df.index.month_name()

ts_df.info()










#%%

#Montlhly Average energy consumption
plot_df = ts_df[~ts_df.year.isin([2011, 2014])].groupby(["year", "month_name"])[['energy_consumption',"temperature"]].mean().reset_index()


fig = px.line(plot_df, x="month_name", y='energy_consumption', color="year", line_dash="year", title="Seasonal Plot - Monthly")
fig = format_plot(fig, ylabel="Energy Consumption", xlabel="Month")
fig.update_layout(legend=dict(
                font=dict(size=15),
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
            ),
            yaxis=dict(
                # title_text=ylabel,
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            ),
            xaxis=dict(
                # title_text=xlabel,
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            ))
fig.write_image("../imgs/chapter_3/seasonal_plot_monthly.png")
fig.show()


fig = plotting_utils.multiple_line_plot_secondary_axis(plot_df, 
    x="month_name", 
    primary='energy_consumption', 
    secondary='temperature', 
    color_or_linetype="year", 
    title="Seasonal Plot Monthly: Multivariate",
    use_linetype=True,
    greyscale=False
)
fig = format_plot(fig, ylabel="Energy Consumption", xlabel="Month")
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=0.9,
    xanchor="right",
                x=1,
))
fig.write_image("../imgs/chapter_3/seasonal_plot_monthly_mv.png")
fig.show()

#%%

ts_df.info()

from imputation.interpolation import SeasonalInterpolation
ts = SeasonalInterpolation(seasonal_period=48*7).fit_transform(ts_df.energy_consumption.values.reshape(-1,1)).squeeze()



from statsmodels.tsa.seasonal import seasonal_decompose
from decomposition.seasonal import STL, FourierDecomposition, MultiSeasonalDecomposition

from plotly.subplots import make_subplots
import plotly.graph_objects as go
def decomposition_plot(
        ts_index, observed=None, seasonal=None, trend=None, resid=None
    ):
        """Plots the decomposition output
        """
        series = []
        if observed is not None:
            series += ["Original"]
        if trend is not None:
            series += ["Trend"]
        if seasonal is not None:
            series += ["Seasonal"]
        if resid is not None:
            series += ["Residual"]
        if len(series) == 0:
            raise ValueError(
                "All component flags were off. Need atleast one of the flags turned on to plot."
            )
        fig = make_subplots(
            rows=len(series), cols=1, shared_xaxes=True, subplot_titles=series
        )
        x = ts_index
        row = 1
        if observed is not None:
            fig.append_trace(
                go.Scatter(x=x, y=observed, name="Original"), row=row, col=1
            )
            row += 1
        if trend is not None:
            fig.append_trace(
                go.Scatter(x=x, y=trend, name="Trend"), row=row, col=1
            )
            row += 1
        if seasonal is not None:
            fig.append_trace(
                go.Scatter(x=x, y=seasonal, name="Seasonal"),
                row=row,
                col=1,
            )
            row += 1
        if resid is not None:
            fig.append_trace(
                go.Scatter(x=x, y=resid, name="Residual"), row=row, col=1
            )
            row += 1

        fig.update_layout(
            title_text="Seasonal Decomposition",
            autosize=False,
            width=1200,
            height=700,
            title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
            titlefont={"size": 20},
            legend_title=None,
            showlegend=False,
            legend=dict(
                font=dict(size=15),
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
            ),
            yaxis=dict(
                # title_text=ylabel,
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            ),
            xaxis=dict(
                # title_text=xlabel,
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            )
        )
        return fig

res = seasonal_decompose(ts, period=7*48, model="additive", extrapolate_trend="freq", filt=np.repeat(1/(30*48), 30*48))

fig = decomposition_plot(ts_df.index, res.observed, res.seasonal, res.trend, res.resid)
fig.write_image("../imgs/chapter_3/moving_avg_decomposition.png")
fig.show(renderer="svg")


ts_df.info()
ts_df.year.unique()

ts_df2 = ts_df[ts_df.year==2013]
ts_df2.info()


stl = STL(seasonality_period=7*48, model = "additive")
res_new = stl.fit(ts_df.energy_consumption)





fig = res_new.plot(interactive=True)
fig.update_layout(
            legend=dict(
                font=dict(size=15),
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
            ),
            yaxis=dict(
                # title_text=ylabel,
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            ),
            xaxis=dict(
                # title_text=xlabel,
                titlefont=dict(size=15),
                tickfont=dict(size=15),
            )
        )
fig.write_image("../imgs/chapter_3/stl_decomposition.png")
fig.show(renderer="svg")


fig.update_xaxes(type="date", range=["2012-11-4", "2012-11-11"])
fig.write_image("../imgs/chapter_3/stl_decomposition_zoomed.png")
fig.show(renderer="svg")






