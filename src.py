#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import math
import statistics

PATH = "/home/noellie/Documents/Suivi saison 2023/bayer_melon/"
FILE = "test_CU2304-00086.feather"
METEO = "meteo.csv"
IRRIGATION = 'irrigation.csv'

#%% Fonctions
def groups(x):
    if x == "VS823Cx-00668" or x == "VS823Cx-00014" or x == "VS823Cx-00072":
        return 'hydroscore'
    elif x == "VS823Cx-00546" or x == "VS823Cx-00647" or x == "VS823Cx-00422" :
        return 'stress'
    elif x == "VS823Cx-00572" or x == "VS823Cx-00230" or x == "VS823Cx-00527" :
        return 'temoin'
    
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

def make_ks_test(datacube_1,datacube_2):

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
            try:
                result = stats.kstest(sample_0, sample_1)
                stat_matrix[i, h] = result.statistic
            except:
                stat_matrix[i, h] = np.nan
    return stat_matrix

def make_datacube(df):
    n_days = (df['date'].max() - df['date'].min()).days + 1
    n_points_max = 60
    datacube = np.empty((n_days, 24, n_points_max))
    datacube[:] = np.nan
    date_grid = df['date'].drop_duplicates().sort_values().tolist()
    sample_sizes = []
    for i, d in enumerate(date_grid):
        for h in range(24):
            select = (df['date'] == d) & (df['hour'] == h)
            sample = df.loc[select, 'r2'].values
            sample_sizes.append(len(sample))
            datacube[i, h, :len(sample)] = sample

    datacube = datacube[:,:,:max(sample_sizes)]
    start_date = min(date_grid)
    end_date = max(date_grid)
    return datacube, start_date, end_date

#%% Chargement des fichiers
raw_df = pd.read_feather(f'{PATH}{FILE}')
meteo = pd.read_csv(f'{PATH}{METEO}')
irrigation = pd.read_csv(f'{PATH}{IRRIGATION}')

#%% Préparation des fichiers et médiane interchannel
df = raw_df
df['hour'] = df['datetime'].dt.hour.astype(str)
df['date'] = df['datetime'].dt.date.astype(str)
combined_strings = [f'{date} {time}:00:00' for date, time in zip(df['date'], df['hour'])]
df['datetime'] = pd.to_datetime(combined_strings, format='mixed')
df['modality'] = df['device_sn'].apply(groups)
df['hour'] = df['datetime'].dt.hour.astype(float)
df = df.loc[df['date'] > "2023-07-09"]
df = df.loc[df['date'] < "2023-08-17"]
df = df.loc[df['device_sn'] != "VS823Cx-00647"]
df = df.loc[df['device_sn'] != "VS823Cx-00422"]
df['mobilisation'] = df['spectral_power_law_exponent'] * -1
df = df.reset_index(drop=True)
df = df.sort_values(['device_sn','datetime'])

df = df.groupby(['modality','device_sn','date','hour','datetime']).median(numeric_only=True).reset_index()


meteo['date'] = pd.to_datetime(meteo['date'], format='mixed')

irrigation['date'] = pd.to_datetime(irrigation['date'], format='mixed')
irrigation['datetime'] = pd.to_datetime(irrigation['datetime'], format='mixed')

#%% Plot par device ou modality avec pluie et irrigation
colors = ["#FF0000","#ff8000","#ffbf00",
          '#ffff00','#80ff00',"#00ffff",
          "#0080ff","#0000ff","#bf00ff"]

fig = make_subplots(rows=3,
                    cols = 1, 
                    shared_xaxes=True,
                    vertical_spacing=0.06)
i = 0

AGG = "device"

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



        i = i+1

fig.add_trace(go.Bar(
            x=meteo["date"],
            y=meteo["rainfall"],
            marker=dict(color='blue'),
            name="rainfall"),
            row=3, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_temoin"],
            marker=dict(color='light blue'),
            name="irrigation_temoin"),
            row=3, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_stress"],
            marker=dict(color='red'),
            name="irrigation_stress"),
            row=3, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_hydroscore"],
            marker=dict(color='green'),
            name="irrigation_hydroscore"),
            row=3, col=1)

fig.update_layout(
    yaxis=dict(title="mobilisation", range = [0,3]),
    yaxis2=dict(title="r2", range = [0,1]),
    legend=dict(title=AGG),
    )  

fig.show()
fig.write_html(f'{PATH}{FILE}_by_{AGG}.html')

#%% Boxplot par modality

fig = px.box(df,
             x='hour',
             y='mobilisation',
             color='modality',
             points = False)
fig.show()

#%% Quelques stats par modality
infos = pd.DataFrame()
for moda in df['modality'].unique():
    print(moda)
    df_moda = df.loc[df['modality'] == moda]
    infos_ = pd.DataFrame()
    infos_['modality'] = moda,
    infos_['moy'] = df_moda.groupby(['modality']).mean(numeric_only=True).reset_index()['spectral_power_law_exponent'][0],
    infos_ ['min'] = df_moda.groupby(['modality']).min(numeric_only=True).reset_index()['spectral_power_law_exponent'][0],
    infos_ ['max'] = df_moda.groupby(['modality']).max(numeric_only=True).reset_index()['spectral_power_law_exponent'][0]
    
    infos = pd.concat([infos,infos_])

print(infos) 

# %% Séparation des modalities en 3 dataframes et heatmap par jour et heure
df_stress = df.loc[df['modality'] == 'stress']
df_hydroscore = df.loc[df['modality'] == 'hydroscore']
df_temoin = df.loc[df['modality'] == 'temoin']

fig = go.Figure(data =
     go.Heatmap(x = df_temoin['date'],
                y = df_temoin['hour'], 
                z = df_temoin['mobilisation'], 
                colorscale='thermal',
                zmin = 0.5,
                zmax = 3))

fig.update_layout(title = "temoin")
fig.show()

fig = go.Figure(data =
     go.Heatmap(x = df_hydroscore['date'],
                y = df_hydroscore['hour'], 
                z = df_hydroscore['mobilisation'], 
                colorscale='thermal',
                zmin = 0.5,
                zmax = 3))

fig.update_layout(title = "hydroscore")
fig.show()

fig = go.Figure(data =
     go.Heatmap(x = df_stress['date'],
                y = df_stress['hour'], 
                z = df_stress['mobilisation'], 
                colorscale='thermal',
                zmin = 0.5,
                zmax = 3))

fig.update_layout(title = "stress")
fig.show()

# %% Mise en forme des raw_data
raw_df['modality'] = raw_df['device_sn'].apply(groups)
raw_df['date'] = raw_df['datetime'].dt.date
raw_df['hour'] = raw_df['datetime'].dt.hour
raw_df['date'] = pd.to_datetime(raw_df['date']).dt.date
raw_df = raw_df.loc[raw_df['date'] > dt.date(2023, 7, 9)]
raw_df = raw_df.loc[raw_df['date'] < dt.date(2023, 8, 17)]
raw_df = raw_df.loc[raw_df['device_sn'] != "VS823Cx-00647"]
raw_df = raw_df.loc[raw_df['device_sn'] != "VS823Cx-00422"]

raw_df['mobilisation'] = raw_df['spectral_power_law_exponent'] * -1
raw_df['r2'] = raw_df['spectral_power_law_goodness_of_fit']

# %% Raw_data par moda
raw_df_stress = raw_df.loc[raw_df['modality'] == 'stress']
raw_df_hydroscore = raw_df.loc[raw_df['modality'] == 'hydroscore']
raw_df_temoin = raw_df.loc[raw_df['modality'] == 'temoin']

# %% Faire datacube par moda
datacube_temoin, _, _ = make_datacube(raw_df_temoin)
datacube_hydroscore, _, _ = make_datacube(raw_df_hydroscore)
datacube_stress, _, _ = make_datacube(raw_df_stress)


# %% Test de kruskal wallis
result_hydro_kw = make_kw_test(datacube_temoin,datacube_hydroscore)
result_stress_kw = make_kw_test(datacube_temoin,datacube_stress)

# %% Test de kulgomorof smirnov
result_hydro_ks = make_ks_test(datacube_temoin,datacube_hydroscore)
result_stress_ks = make_ks_test(datacube_temoin,datacube_stress)

# %% Viz cubes
RESULT_TO_PLOT_1 = result_hydro_kw
name_1 = "result_hydro_kw"
RESULT_TO_PLOT_2 = result_stress_kw
name_2 = "result_stress_kw"

date_grid = df['date'].drop_duplicates().sort_values().tolist()

fig = go.Figure(data =
     go.Heatmap(x = list(range(24)),
                y = date_grid, 
                z = RESULT_TO_PLOT_1, 
                colorscale='thermal',
                zmin = -10,
                zmax = 0))

fig.update_layout(title = f'KW statistic : {name_1}')
fig.show()

fig2 = go.Figure(data =
     go.Heatmap(x = list(range(24)),
                y = date_grid, 
                z = RESULT_TO_PLOT_2, 
                colorscale='thermal',
                zmin = -10,
                zmax = 0))

fig2.update_layout(title = f'KW statistic : {name_2}')
fig2.show()

fig.write_html(f'{PATH}_{name_1}_datacube.html')
fig2.write_html(f'{PATH}_{name_2}_datacube.html')

# %% Moyenne des statistics par jour
mean_result_hydro_kw = np.mean(result_hydro_kw, axis=1)
mean_result_stress_kw = np.mean(result_stress_kw, axis=1)

mean_result_hydro_ks = np.mean(result_hydro_ks, axis=1)
mean_result_stress_ks = np.mean(result_stress_ks, axis=1)
#%% Plot evolution de la feat et de la statistique de kruskall wallis sur cette feat 
fig = make_subplots(rows=3,
                    cols = 1, 
                    shared_xaxes=True,
                    vertical_spacing=0.06)

for moda in df['modality'].unique() : 

    df_moda = df.loc[df['modality'] == moda]

    fig.add_trace(go.Box(x=df_moda['date'],
                        y=df_moda['mobilisation'],
                        name=moda,
                        boxpoints = False),
                        row=1,col=1)
    

fig.add_trace(go.Scatter(
                x=date_grid,
                y=mean_result_hydro_kw,
                name ='temoin vs hydro'),
                row=2,col=1)


fig.add_trace(go.Scatter(
                x=date_grid,
                y=mean_result_stress_kw,
                name ='temoin vs stress'),
                row=2,col=1)

fig.add_trace(go.Bar(
            x=meteo["date"],
            y=meteo["rainfall"],
            marker=dict(color='blue'),
            name="rainfall"),
            row=3, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_temoin"],
            marker=dict(color='light blue'),
            name="irrigation_temoin"),
            row=3, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_stress"],
            marker=dict(color='red'),
            name="irrigation_stress"),
            row=3, col=1)

fig.add_trace(go.Bar(
            x=irrigation["datetime"],
            y=irrigation["irrigation_hydroscore"],
            marker=dict(color='green'),
            name="irrigation_hydroscore"),
            row=3, col=1)

fig.add_hline(y=-3, line_width=1, line_dash="dash", line_color="red",row=2,col=1)
                

fig.update_layout(boxmode='group') 
fig.show()
fig.write_html(f'{PATH}mobilisation_statistique_kw.html')


# %%
