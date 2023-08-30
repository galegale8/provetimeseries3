
#%%
import pandas as pd



df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx", skiprows=1)

df.info()

pd.to_datetime("4|1|1987", format="%d|%m|%Y").strftime("%d, %B %Y")

df['date2'] = pd.to_datetime(df['date'], yearfirst=True)

df.date.min(),df.date.max()

df.date.iloc[0]

df.date.dt.day_of_year.iloc[0]

df.date.dt.dayofweek.iloc[0]

df.set_index("date", inplace=True)

df["2010-01-04": "2010-02-06"]

pd.date_range(start="2018-01-20", end="2018-01-23", freq="D").astype(str).tolist()






# %%
