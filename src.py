# %%

import psycopg2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import math
import statistics

# %%

year = 2023
operation = "CU2305-00092"
INPUT_DIR = f"/media/noellie/HDD/files/{year}/{operation}"
OUTPUT_DIR = f"/home/noellie/Documents/Suivi saison 2023/lafite"
FILE = f"{operation}.feather"
START_DATE = dt.date(2023, 6, 9)
END_DATE = dt.date(2023, 9, 20)

# %%

def get_general_info():
    connection = psycopg2.connect(host="localhost", port="5432", dbname="agro_weather", user="postgres", password="123456")
    connection.autocommit = True
    sql_general_information = f"""SELECT *
        FROM general_information
        WHERE year = {year} and operation = '{operation}'"""
    df_general_information = pd.read_sql(sql_general_information , con=connection)
    return df_general_information

# %%

raw_df = pd.read_feather(f'{INPUT_DIR}/{FILE}')
raw_df

# %%

def prep_df(df_general_information, raw_df, start_date, end_date):
    df = raw_df.merge(df_general_information)
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date
    df['mobilisation'] = df['spectral_power_law_exponent'] * -1
    df['r2'] = raw_df['spectral_power_law_goodness_of_fit']
    df = df.loc[df['date'] >= start_date]
    df = df.loc[df['date'] <= end_date]
    df = df.reset_index(drop=True)
    df = df.sort_values(['device_sn','datetime'])
    return df


def make_datacube(df, feat):
    n_days = (df['date'].max() - df['date'].min()).days + 1
    n_points_max = 60
    datacube = np.empty((n_days, 24, n_points_max))
    datacube[:] = np.nan
    date_grid = df['date'].drop_duplicates().sort_values().tolist()
    sample_sizes = []
    for i, d in enumerate(date_grid):
        for h in range(24):
            select = (df['date'] == d) & (df['hour'] == h)
            sample = df.loc[select, feat].values
            sample_sizes.append(len(sample))
            datacube[i, h, :len(sample)] = sample

    datacube = datacube[:,:,:max(sample_sizes)]
    start_date = min(date_grid)
    end_date = max(date_grid)
    return datacube, start_date, end_date

def make_test(datacube_1, datacube_2, type):

    n_days = len(datacube_1)
    stat_matrix = np.empty((n_days, 24))
    pval_matrix = np.empty((n_days, 24))
    stat_matrix[:] = np.nan
    pval_matrix[:] = np.nan
    date_grid = df['date'].drop_duplicates().sort_values().tolist()

    for i, d in enumerate(date_grid):
        for h in range(24):
            sample_0 = datacube_1[i,h,:]
            sample_0 = sample_0[~np.isnan(sample_0)]
            sample_1 = datacube_2[i,h,:]
            sample_1 = sample_1[~np.isnan(sample_1)]
            if type == 'kw':
                result = stats.kruskal(sample_0, sample_1)
                stat_matrix[i, h] = result.statistic
                pval_matrix[i, h] = math.log(result.pvalue)
            elif type == 'ks':
                try:
                    result = stats.kstest(sample_0, sample_1)
                    stat_matrix[i, h] = result.statistic
                    pval_matrix[i, h] = math.log(result.pvalue)
                except:
                    stat_matrix[i, h] = np.nan
                    pval_matrix[i, h] = np.nan
            else:
                raise Exception('Unknown test!')

    return stat_matrix, pval_matrix


# %%


df_general_information = get_general_info()
raw_df = pd.read_feather(f'{INPUT_DIR}/{FILE}')
df = prep_df(df_general_information, raw_df, START_DATE, END_DATE)
df

# %%

def run_all_tests(df, modalities):
    res_dict = {k:{} for k in modalities}
    for i, moda_i in enumerate(modalities):
        if i == len(modalities):
            break
        for j in range(i + 1, len(modalities)):
            moda_j = modalities[j]
            print(moda_i, moda_j)
            df_i = df.loc[df['modality'] == moda_i]
            df_j = df.loc[df['modality'] == moda_j]
            datacube_mob_i, _, _ = make_datacube(df_i, 'mobilisation')
            datacube_mob_j, _, _ = make_datacube(df_j, 'mobilisation')
            datacube_r2_i, _, _ = make_datacube(df_i, 'r2')
            datacube_r2_j, _, _ = make_datacube(df_j, 'r2')
            stat_mob_kw_ij, pval_mob_kw_ij = make_test(datacube_mob_i, datacube_mob_j, 'kw')
            stat_mob_ks_ij, pval_mob_ks_ij = make_test(datacube_mob_i, datacube_mob_j, 'ks')
            stat_r2_kw_ij, pval_r2_kw_ij = make_test(datacube_mob_i, datacube_mob_j, 'kw')
            stat_r2_ks_ij, pval_r2_ks_ij = make_test(datacube_r2_i, datacube_r2_j, 'ks')
            res_dict[moda_i] = {moda_j: {
                'mobilisation': {
                    'kw': {'stat': stat_mob_kw_ij, 'pval': pval_mob_kw_ij},
                    'ks': {'stat': stat_mob_ks_ij, 'pval': pval_mob_ks_ij}
                },
                'r2': {
                    'kw': {'stat': stat_r2_kw_ij, 'pval': pval_r2_kw_ij},
                    'ks': {'stat': stat_r2_ks_ij, 'pval': pval_r2_ks_ij}
                }}}
    return res_dict

# %%

def make_viz_from_res(date_grid, res, df, feat, type, stat_or_pval, moda_i, moda_j):
    title = f'{feat} | {type} {stat_or_pval} | {moda_i} vs {moda_j}'
    fig = go.Figure(data =
        go.Heatmap(x = list(range(24)),
        y = date_grid, 
        z = res, 
        colorscale='thermal',
        zmin = np.min(res),
        zmax = np.max(res)))
    fig.update_layout(title = title)
    fig.show()
    fig.write_html(f'{OUTPUT_DIR}/heatmap | {title}.html')
    #
    fig = make_subplots(
        rows=2,
        cols = 1, 
        shared_xaxes=True,
        vertical_spacing=0.06
    )
    for moda in [moda_i, moda_j]:
        df_moda = df.loc[df['modality'] == moda]
        fig.add_trace(go.Box(
            x=df_moda['date'],
            y=df_moda[feat],
            name=moda,
            boxpoints = False),
            row=1,col=1)
    fig.add_trace(go.Scatter(
                    x=date_grid,
                    y=np.nanmean(res, axis=1),
                    name ='couvert vs travail'),
                    row=2,col=1)
    fig.update_layout(boxmode='group') 
    if stat_or_pval == 'pval':
        fig.add_hline(y=-3, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.show()
    fig.write_html(f'{OUTPUT_DIR}/timeseries | {title}.html')


def make_viz_for_one_contrast(df, res_dict, moda_i, moda_j):
    df_ij = df.loc[df['modality'].isin([moda_i, moda_j])]
    date_grid = df_ij['date'].drop_duplicates().sort_values().tolist()
    res_dict_ij = res_dict[moda_i][moda_j]
    mob_kw_stat = res_dict_ij['mobilisation']['kw']['stat']
    mob_kw_pval = res_dict_ij['mobilisation']['kw']['pval']
    mob_ks_stat = res_dict_ij['mobilisation']['ks']['stat']
    mob_ks_pval = res_dict_ij['mobilisation']['ks']['pval']
    r2_kw_stat = res_dict_ij['r2']['kw']['stat']
    r2_kw_pval = res_dict_ij['r2']['kw']['pval']
    r2_ks_stat = res_dict_ij['r2']['ks']['stat']
    r2_ks_pval = res_dict_ij['r2']['ks']['pval']
    make_viz_from_res(date_grid, mob_kw_stat, df_ij, 'mobilisation', 'KW', 'stat', moda_i, moda_j)
    make_viz_from_res(date_grid, mob_kw_pval, df_ij, 'mobilisation', 'KW', 'pval', moda_i, moda_j)
    make_viz_from_res(date_grid, mob_ks_stat, df_ij, 'mobilisation', 'KS', 'stat', moda_i, moda_j)
    make_viz_from_res(date_grid, mob_ks_pval, df_ij, 'mobilisation', 'KS', 'pval', moda_i, moda_j)
    make_viz_from_res(date_grid, r2_kw_stat, df_ij, 'r2', 'KW', 'stat', moda_i, moda_j)
    make_viz_from_res(date_grid, r2_kw_pval, df_ij, 'r2', 'KW', 'pval', moda_i, moda_j)
    make_viz_from_res(date_grid, r2_ks_stat, df_ij, 'r2', 'KS', 'stat', moda_i, moda_j)
    make_viz_from_res(date_grid, r2_ks_pval, df_ij, 'r2', 'KS', 'pval', moda_i, moda_j)

def make_all_contrast_viz(df):
    modalities = list(set(df['modality']))
    res_dict = run_all_tests(df, modalities)
    for i, moda_i in enumerate(modalities):
        if i == len(modalities):
            break
        for j in range(i + 1, len(modalities)):
            moda_j = modalities[j]
            print(moda_i, moda_j)
            make_viz_for_one_contrast(df, res_dict, moda_i, moda_j)

make_all_contrast_viz(df)
# %%
