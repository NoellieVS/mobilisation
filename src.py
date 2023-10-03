#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import plotly.express as px
from plotly.subplots import make_subplots

PATH = ""
FILE = "test_CU2304-00086.feather"

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

#%%
colors = ["#FF0000","#ff8000","#ffbf00",
          '#ffff00','#80ff00',"#00ffff",
          "#0080ff","#0000ff","#bf00ff"]

fig = make_subplots(rows=2,
                    cols = 1, 
                    shared_xaxes=True,
                    vertical_spacing=0.06)
i = 0

for dev in df['device_sn'].unique():

    df_dev = df.loc[df['device_sn']==dev].reset_index(inplace = False)
    
    fig.add_trace(go.Scatter(
                    x=df_dev["datetime"],
                    y=df_dev["spectral_power_law_exponent"],
                    mode='markers',
                    marker = dict(color=colors[i]),
                    legendgroup=df_dev['modality'][0],
                    name=dev,
                    showlegend=False),
                    row=1, col=1)

    fig.add_trace(go.Scatter(
                    x=df_dev["datetime"],
                    y=df_dev["spectral_power_law_goodness_of_fit"],
                    mode='markers',
                    marker = dict(color=colors[i]),
                    name=dev,
                    legendgroup=df_dev['modality'][0]),
                    row=2, col=1)
    
    fig.update_layout(
        yaxis=dict(title="spectral_power_law_exponent", range = [-4,0]),
        yaxis2=dict(title="spectral_power_law_goodness_of_fit", range = [0,1.2]),
        legend=dict(title="modality"),
        )      
    
    i = i+1

fig.show()
# fig.write_html(f'{PATH}{FILE}.html')
#%%

fig = px.scatter(df,
             x='datetime',
             y='spectral_power_law_exponent',
             color='modality')
            #  points = False)
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
fig = go.Figure(data =
     go.Heatmap(x = df['hour'],
                y = df['modality'], 
                z = df['spectral_power_law_exponent'], 
                colorscale='thermal',
                zmin = -2.8,
                zmax = -1.3))

fig.update_layout(title = "spectral_power_law_exponent")
fig.show()
# %%
