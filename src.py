#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import math


PATH = ""
FILE = "test_CU2304-00086.feather"
METEO = "meteo.csv"
IRRIGATION = 'irrigation.csv'

def groups(x):
    if x == "VS823Cx-00668" or x == "VS823Cx-00014" or x == "VS823Cx-00072":
        return 'hydroscore'
    elif x == "VS823Cx-00546" or x == "VS823Cx-00647" or x == "VS823Cx-00422" :
        return 'stress'
    elif x == "VS823Cx-00572" or x == "VS823Cx-00230" or x == "VS823Cx-00527" :
        return 'temoin'
    
#%%
df = pd.read_feather(f'{PATH}{FILE}')

df['hour'] = df['datetime'].dt.hour.astype(str)
df['date'] = df['datetime'].dt.date.astype(str)
combined_strings = [f'{date} {time}:00:00' for date, time in zip(df['date'], df['hour'])]
df['datetime'] = pd.to_datetime(combined_strings, format='mixed')
df = df.groupby(['device_sn','date','hour','datetime']).median(numeric_only=True).reset_index()
df['hour'] = df['datetime'].dt.hour.astype(float)
df['modality'] = df['device_sn'].apply(groups)
df = df.loc[df['date'] > "2023-07-09"]
df = df.loc[df['date'] < "2023-08-22"]
df = df.loc[df['device_sn'] != "VS823Cx-00647"]
df['mobilisation'] = df['spectral_power_law_exponent'] * -1
df = df.reset_index(drop=True)
df = df.sort_values(['device_sn','datetime'])
    
#%%
for i in range(0,len(df)):
    if (i-1 > -1):
        df["derive"][i] = df['mobilisation'][i] - df['mobilisation'][i-1]
    else : 
        df["derive"] = None

#%%
meteo = pd.read_csv(f'{PATH}{METEO}')
meteo['date'] = pd.to_datetime(meteo['date'], format='mixed')

irrigation = pd.read_csv(f'{PATH}{IRRIGATION}')
irrigation['date'] = pd.to_datetime(irrigation['date'], format='mixed')
irrigation['datetime'] = pd.to_datetime(irrigation['datetime'], format='mixed')

#%% 
colors = ["#FF0000","#ff8000","#ffbf00",
          '#ffff00','#80ff00',"#00ffff",
          "#0080ff","#0000ff","#bf00ff"]

fig = make_subplots(rows=4,
                    cols = 1, 
                    shared_xaxes=True,
                    vertical_spacing=0.06)
i = 0

AGG = "modality"

if AGG == "device" :

    for dev in df['device_sn'].unique():

        df_viz = df.loc[df['device_sn']==dev].reset_index(inplace = False)
        df_viz = df_viz.sort_values(['datetime'])
      
        fig.add_trace(go.Scatter(
                        x=df_viz["datetime"],
                        y=df_viz["mobilisation"],
                        mode='lines',
                        marker = dict(color=colors[i]),
                        legendgroup=dev,
                        name=dev,
                        showlegend=False),
                        row=1, col=1)

        fig.add_trace(go.Scatter(
                        x=df_viz["datetime"],
                        y=df_viz["spectral_power_law_goodness_of_fit"],
                        mode='lines',
                        marker = dict(color=colors[i]),
                        name=dev,
                        legendgroup=dev),
                        row=2, col=1)  

        fig.add_trace(go.Scatter(
                        x=df_viz["datetime"],
                        y=df_viz["derive"],
                        mode='lines',
                        marker = dict(color=colors[i]),
                        name=dev,
                        legendgroup=dev,
                        showlegend=False),
                        row=3, col=1)  

        i = i+1

else : 


    for moda in df['modality'].unique():

        df_viz = df.loc[df['modality']==moda].reset_index(inplace = False)
        df_viz = df_viz.groupby(['modality','date','hour','datetime']).mean(numeric_only=True).reset_index()
        df_viz = df_viz.sort_values(['datetime'])
      
        fig.add_trace(go.Scatter(
                        x=df_viz["datetime"],
                        y=df_viz["mobilisation"],
                        mode='lines',
                        marker = dict(color=colors[i]),
                        legendgroup=moda,
                        name=moda,
                        showlegend=False),
                        row=1, col=1)

        fig.add_trace(go.Scatter(
                        x=df_viz["datetime"],
                        y=df_viz["spectral_power_law_goodness_of_fit"],
                        mode='lines',
                        marker = dict(color=colors[i]),
                        name=moda,
                        legendgroup=moda),
                        row=2, col=1)  

        fig.add_trace(go.Scatter(
                        x=df_viz["datetime"],
                        y=df_viz["derive"],
                        mode='lines',
                        marker = dict(color=colors[i]),
                        name=moda,
                        legendgroup=moda,
                        showlegend=False),
                        row=3, col=1)  


        i = i+1

fig.add_trace(go.Bar(
            x=meteo["date"],
            y=meteo["rainfall"],
            marker=dict(color='blue'),
            name="rainfall"),
            row=4, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_temoin"],
            marker=dict(color='light blue'),
            name="irrigation_temoin"),
            row=4, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_stress"],
            marker=dict(color='red'),
            name="irrigation_stress"),
            row=4, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_hydroscore"],
            marker=dict(color='green'),
            name="irrigation_hydroscore"),
            row=4, col=1)

fig.update_layout(
    yaxis=dict(title="mobilisation", range = [0,3]),
    yaxis2=dict(title="r2", range = [0,1]),
    yaxis3=dict(title="dérivé", range = [-2,2]),
    legend=dict(title="modality"),
    )  

fig.show()
fig.write_html(f'{PATH}{FILE}.html')
#%%

fig = px.box(df,
             x='hour',
             y='spectral_power_law_exponent',
             color='modality',
             points = False)
fig.show()
#%%
stats = pd.DataFrame()
for moda in df['modality'].unique():
    print(moda)
    df_moda = df.loc[df['modality'] == moda]
    stats_ = pd.DataFrame()
    stats_['modality'] = moda,
    stats_['moy'] = df_moda.groupby(['modality']).mean(numeric_only=True).reset_index()['spectral_power_law_exponent'][0],
    stats_['min'] = df_moda.groupby(['modality']).min(numeric_only=True).reset_index()['spectral_power_law_exponent'][0],
    stats_['max'] = df_moda.groupby(['modality']).max(numeric_only=True).reset_index()['spectral_power_law_exponent'][0]
    
    stats = pd.concat([stats,stats_])

print(stats) 

# %%
df_stress = df.loc[df['modality'] == 'stress']
df_hydroscore = df.loc[df['modality'] == 'hydroscore']
df_temoin = df.loc[df['modality'] == 'temoin']

fig = go.Figure(data =
     go.Heatmap(x = df_temoin['hour'],
                y = df_temoin['date'], 
                z = df_temoin['mobilisation'], 
                colorscale='thermal',
                zmin = 0.5,
                zmax = 3))

fig.update_layout(title = "temoin")
fig.show()

# %%

df = pd.read_feather(f'{PATH}{FILE}')

df['modality'] = df['device_sn'].apply(groups)
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['date'] = pd.to_datetime(df['date']).dt.date
df = df.loc[df['date'] > dt.date(2023, 7, 9)]
df = df.loc[df['date'] < dt.date(2023, 8, 22)]
df = df.loc[df['device_sn'] != "VS823Cx-00647"]
df['mobilisation'] = df['spectral_power_law_exponent'] * -1

def make_datacube(df):
    n_days = (df['date'].max() - df['date'].min()).days + 1
    n_points_max = 60
    datacube = np.empty((n_days, 24, n_points_max))
    datacube[:] = np.nan
    date_grid = df['date'].drop_duplicates().sort_values().tolist()
    sample_sizes = []
    for i, d in enumerate(date_grid):
        for h in range(24):
            # print(d, h)
            select = (df['date'] == d) & (df['hour'] == h)
            sample = df.loc[select, 'mobilisation'].values
            sample_sizes.append(len(sample))
            datacube[i, h, :len(sample)] = sample

    datacube = datacube[:,:,:max(sample_sizes)]
    start_date = min(date_grid)
    end_date = max(date_grid)
    return datacube, start_date, end_date

df_temoin = df.loc[df['modality'] == 'temoin']
df_hydroscore = df.loc[df['modality'] == 'hydroscore']
df_stress= df.loc[df['modality'] == 'stress']

datacube_temoin, _, _ = make_datacube(df_temoin)
datacube_hydroscore, _, _ = make_datacube(df_hydroscore)
datacube_stress, _, _ = make_datacube(df_stress)


# %%
def make_kw_test(datacube_1,datacube_2):

    n_days = len(datacube_1)
    stat_matrix = np.empty((n_days, 24))
    stat_matrix[:] = np.nan
    date_grid = df['date'].drop_duplicates().sort_values().tolist()

    for i, d in enumerate(date_grid):
        for h in range(24):
            sample_0 = datacube_1[i,h,:]
            sample_0 = sample_0[~np.isnan(sample_0)]
            sample_1 = datacube_2[i,h,:]
            sample_1 = sample_1[~np.isnan(sample_1)]
            result = stats.kruskal(sample_0, sample_1)
            stat_matrix[i, h] = math.log(result.pvalue)
    return stat_matrix

result = make_kw_test(datacube_temoin,datacube_hydroscore)


# %%
date_grid = df['date'].drop_duplicates().sort_values().tolist()

fig = go.Figure(data =
     go.Heatmap(x = list(range(24)),
                y = date_grid, 
                z = result, 
                colorscale='thermal',
                zmin = -6,
                zmax = -3))

fig.update_layout(title = "temoin vs hydro")
fig.show()
fig.write_html(f'{PATH}_datacube.html')

# %%
