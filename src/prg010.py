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


from utils.data_utils import compact_to_expanded

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


block_df.iloc[0]
exp_block_df.iloc[0]

ts_df = exp_block_df[exp_block_df.LCLid=="MAC000193"].set_index("timestamp")

ts_df.info()



# %%

fig = px.line(ts_df, y="energy_consumption", title="Energy Consumption for MAC000193")
fig = format_plot(fig, ylabel="Energy Consumption")
fig.write_image("../imgs/chapter_3/line_plot.png")
fig.show()

ts_df["rolling_monthly_avg"] = ts_df["energy_consumption"].rolling(window=48*30).mean()

fig = px.line(ts_df, y="rolling_monthly_avg", title="Rolling Monthly Average Energy Consumption for MAC000193")
fig = format_plot(fig, ylabel="Energy Consumption")
fig.write_image("../imgs/chapter_3/rolling_avg_line_plot.png")
fig.show()


zoom_window = slice("2012-03-16", "2012-03-30")
fig = plotting_utils.two_line_plot_secondary_axis(x=ts_df[zoom_window].index, 
                                   y1=ts_df[zoom_window].energy_consumption, 
                                   y2=ts_df[zoom_window].temperature, 
                                   y1_name="Energy Consumption", 
                                   y2_name="Temperature", 
                                   title="Temperature and Energy Consumption (2012-03-16 to 2012-03-30)",
                                   greyscale=False,
                                   dash_secondary=True)
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
fig.write_image("../imgs/chapter_3/temp_vs_energy_zoomed.png")
fig.show()


ts_df["year"] = ts_df.index.year
ts_df["month_name"] = ts_df.index.month_name()
plot_df = ts_df[~ts_df.year.isin([2011, 2014])].groupby(["year", "month_name"])[['energy_consumption', "temperature"]].mean().reset_index()

plot_df['year_month'] = plot_df["year"].astype(str) +" "+ plot_df['month_name'].astype(str).str.zfill(2)
fig = plotting_utils.two_line_plot_secondary_axis(x=[plot_df.year, plot_df.month_name], y1=plot_df.energy_consumption, y2=plot_df.temperature, y1_name="Energy Consumption", y2_name="Temperature", title="Temperature and Energy Consumption - Monthly", dash_secondary=True)
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
fig.write_image("../imgs/chapter_3/temp_vs_energy_monthly.png")
fig.show()

# %%

import pandas as pd

# Create a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}

dfmy = pd.DataFrame(data)

# Print the DataFrame
new_row = {'Name': 'Piero', 'Age': 40, 'City': 'pescara'}
dfmy = pd.concat([dfmy, pd.DataFrame([new_row])], ignore_index=True)



new_row = {'Name': 'Fabio', 'Age': 30, 'City': 'roma'}
dfmy = pd.concat([dfmy, pd.DataFrame([new_row])], ignore_index=True)



print(dfmy)


import pandas as pd

# Create a sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Value': [10, 15, 20, 25, 30, 35]
}

dfmy = pd.DataFrame(data)

# Group by 'Category' and aggregate 'Value' into a list
result = dfmy.groupby('Category')['Value'].agg(list).reset_index()

# Rename the column
result = result.rename(columns={'Value': 'Values'})

# Print the result
print(result)







# %%
