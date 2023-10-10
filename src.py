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
operation = "CU2304-00086"
in_db = False


INPUT_DIR = f"/media/noellie/HDD/files/{year}/{operation}"
OUTPUT_DIR = f"/home/noellie/Documents/Suivi saison 2023/bayer_melon"
FILE = f"{operation}.feather"
GENERAL_INFO = "general_info.csv"
START_DATE = dt.date(2023, 7, 10)
END_DATE = dt.date(2023, 8, 17)

dev_to_remove = ["VS823Cx-00647","VS823Cx-00422"]

# %%
def get_general_info():
    connection = psycopg2.connect(host="localhost", port="5432", dbname="agro_weather", user="postgres", password="123456")
    connection.autocommit = True
    sql_general_information = f"""SELECT *
        FROM general_information
        WHERE year = {year} and operation = '{operation}'"""
    df_general_information = pd.read_sql(sql_general_information , con=connection)
    return df_general_information
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
    if len(dev_to_remove) > 1:
        df = df[~df['device_sn'].isin(dev_to_remove)]

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
def make_test(datacube_1, start_date_1, end_date_1, datacube_2, start_date_2, end_date_2, type):

    n_days = len(datacube_1)
    stat_matrix = np.empty((n_days, 24))
    pval_matrix = np.empty((n_days, 24))
    stat_matrix[:] = np.nan
    pval_matrix[:] = np.nan

    date_grid_1 = [start_date_1 + dt.timedelta(days=i) for i in range((end_date_1 - start_date_1).days + 1)]
    date_grid_2 = [start_date_2 + dt.timedelta(days=i) for i in range((end_date_2 - start_date_2).days + 1)]
    select_date_1 = [d in date_grid_2 for d in date_grid_1]
    select_date_2 = [d in date_grid_1 for d in date_grid_2]
    start_date = max(start_date_1, start_date_2)
    end_date = min(end_date_1, end_date_2)
    datacube_1 = datacube_1[select_date_1,:,:]
    datacube_2 = datacube_2[select_date_2,:,:]
    if datacube_1.shape[0] != datacube_2.shape[0]:
        raise Exception('Datacubes have a different number of days!')

    for i in range(datacube_1.shape[0]):
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

    return stat_matrix, pval_matrix, start_date, end_date
def run_all_tests(df, modalities):
    res_dict = {k:{} for k in modalities}
    print('Running all tests...')
    for i, moda_i in enumerate(modalities):
        if i == len(modalities):
            break
        res_dict_i = {k:{} for k in modalities}
        for j in range(i + 1, len(modalities)):
            moda_j = modalities[j]
            print(moda_i, moda_j)
            df_i = df.loc[df['modality'] == moda_i]
            df_j = df.loc[df['modality'] == moda_j]
            datacube_mob_i, start_date_mob_i, end_date_mob_i = make_datacube(df_i, 'mobilisation')
            datacube_mob_j, start_date_mob_j, end_date_mob_j = make_datacube(df_j, 'mobilisation')
            datacube_r2_i, start_date_r2_i, end_date_r2_i = make_datacube(df_i, 'r2')
            datacube_r2_j, start_date_r2_j, end_date_r2_j = make_datacube(df_j, 'r2')
            stat_mob_kw_ij, pval_mob_kw_ij, start_date_mob_kw_ij, end_date_mob_kw_ij = make_test(
                datacube_mob_i, start_date_mob_i, end_date_mob_i,
                datacube_mob_j, start_date_mob_j, end_date_mob_j,
                'kw'
            )
            stat_mob_ks_ij, pval_mob_ks_ij, start_date_mob_ks_ij, end_date_mob_ks_ij = make_test(
                datacube_mob_i, start_date_mob_i, end_date_mob_i,
                datacube_mob_j, start_date_mob_j, end_date_mob_j,
                'ks'
            )
            stat_r2_kw_ij, pval_r2_kw_ij, start_date_r2_kw_ij, end_date_r2_kw_ij = make_test(
                datacube_r2_i, start_date_r2_i, end_date_r2_i,
                datacube_r2_j, start_date_r2_j, end_date_r2_j,
                'kw'
            )
            stat_r2_ks_ij, pval_r2_ks_ij, start_date_r2_ks_ij, end_date_r2_ks_ij = make_test(
                datacube_r2_i, start_date_r2_i, end_date_r2_i,
                datacube_r2_j, start_date_r2_j, end_date_r2_j,
                'ks'
            )
            res_dict_i[moda_j] = {
                'mobilisation': {
                    'kw': {
                        'stat': stat_mob_kw_ij,
                        'pval': pval_mob_kw_ij,
                        'start_date': start_date_mob_kw_ij,
                        'end_date': end_date_mob_kw_ij,
                    },
                    'ks': {
                        'stat': stat_mob_ks_ij,
                        'pval': pval_mob_ks_ij,
                        'start_date': start_date_mob_ks_ij,
                        'end_date': end_date_mob_ks_ij
                    }
                },
                'r2': {
                    'kw': {
                        'stat': stat_r2_kw_ij,
                        'pval': pval_r2_kw_ij,
                        'start_date': start_date_r2_kw_ij,
                        'end_date': end_date_r2_kw_ij,
                    },
                    'ks': {
                        'stat': stat_r2_ks_ij,
                        'pval': pval_r2_ks_ij,
                        'start_date': start_date_r2_ks_ij,
                        'end_date': end_date_r2_ks_ij,
                    }
                }
            }
        res_dict[moda_i] = res_dict_i
    return res_dict
def make_viz_from_test(df, res, feat, type, stat_or_pval, moda_i, moda_j):
    start_date = res[moda_i][moda_j][feat][type]['start_date']
    end_date = res[moda_i][moda_j][feat][type]['end_date']
    data = res[moda_i][moda_j][feat][type][stat_or_pval]
    date_grid = [start_date + dt.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    title = f'{feat} | {type} {stat_or_pval} | {moda_i} vs {moda_j}'
    fig = go.Figure(data =
        go.Heatmap(x = list(range(24)),
        y = date_grid, 
        z = data, 
        colorscale='thermal',
        zmin = np.min(data),
        zmax = np.max(data)))
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
                    y=np.nanmean(data, axis=1),
                    name =f'{moda_i} vs {moda_j}'),
                    row=2,col=1)
    fig.update_layout(boxmode='group') 
    if stat_or_pval == 'pval':
        fig.add_hline(y=-3, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.show()
    fig.write_html(f'{OUTPUT_DIR}/timeseries | {title}.html')
def make_viz_for_one_contrast(df, res_dict, moda_i, moda_j):
    df_ij = df.loc[df['modality'].isin([moda_i, moda_j])]
    make_viz_from_test(df_ij, res_dict, 'mobilisation', 'kw', 'stat', moda_i, moda_j)
    make_viz_from_test(df_ij, res_dict, 'mobilisation', 'kw', 'pval', moda_i, moda_j)
    make_viz_from_test(df_ij, res_dict, 'mobilisation', 'ks', 'stat', moda_i, moda_j)
    make_viz_from_test(df_ij, res_dict, 'mobilisation', 'ks', 'pval', moda_i, moda_j)
    make_viz_from_test(df_ij, res_dict, 'r2', 'kw', 'stat', moda_i, moda_j)
    make_viz_from_test(df_ij, res_dict, 'r2', 'kw', 'pval', moda_i, moda_j)
    make_viz_from_test(df_ij, res_dict, 'r2', 'ks', 'stat', moda_i, moda_j)
    make_viz_from_test(df_ij, res_dict, 'r2', 'ks', 'pval', moda_i, moda_j)
def make_all_contrast_viz(df):
    modalities = df['modality'].drop_duplicates().to_list()
    res_dict = run_all_tests(df, modalities)
    print('Making viz...')
    for i, moda_i in enumerate(modalities):
        if i == len(modalities):
            break
        for j in range(i + 1, len(modalities)):
            moda_j = modalities[j]
            print(moda_i, moda_j)
            make_viz_for_one_contrast(df, res_dict, moda_i, moda_j)

# %%
raw_df = pd.read_feather(f'{INPUT_DIR}/{FILE}')

# %%
if in_db == True:
    df_general_information = get_general_info()
else :
    df_general_information = pd.read_csv(f'{OUTPUT_DIR}/{GENERAL_INFO}')

raw_df = pd.read_feather(f'{INPUT_DIR}/{FILE}')

df = prep_df(df_general_information, raw_df, START_DATE, END_DATE)

# %%
modalities = df['modality'].drop_duplicates().to_list()
res_dict = run_all_tests(df, modalities)
# make_viz_from_test(df, res_dict, "mobilisation", "kw", "pval", "M1", "M2")
# make_viz_for_one_contrast(df, res_dict, "M1", "M2")

# %%
make_all_contrast_viz(df)



# %%
