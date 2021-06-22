# %%
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import cluster, preprocessing
from sklearn.model_selection import cross_val_score
from locale import setlocale, LC_ALL, format_string
from requests import Session as sessao
from bz2 import decompress
from json import loads as jsonloads
import folium as fl
setlocale(LC_ALL,'pt_BR.UTF8')
# %%
smp = pd.read_csv("https://github.com/marcio-mutti/data_science_iesb/raw/main/Introducao/Tabela%20de%20refer%C3%AAncia%20cruzada%20-%20UF%201.csv")
smp['pop']=smp['NR_POPULACAO_ESTIMADA'].str.replace('.','').astype(int)
colunas_analise = ['PF_Dados', 'PF_VOZ', 'PF_VOZ_DADOS', 'PJ_Dados', 'PJ_M2M_ESPECIAL', 'PJ_M2M_PADRAO', 'PJ_VOZ', 'PJ_VOZ_DADOS', 'pop']
for col in [x for x in colunas_analise if x != 'pop']:
    smp[f"{col}_dens"] = smp.apply(lambda z: z[col] / z['pop'], axis=1)
uso_colunas_analise = [x+'_dens' for x in colunas_analise if x != 'pop'] + ['pop']
normalizador = preprocessing.QuantileTransformer(output_distribution='normal').fit(smp[uso_colunas_analise])
normalizador = preprocessing.StandardScaler().fit(smp[uso_colunas_analise])
dados_analise = normalizador.transform(smp[colunas_analise])
# %%
estados = dict(zip([11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53],'RO;AC;AM;RR;PA;AP;TO;MA;PI;CE;RN;PB;PE;AL;SE;BA;MG;ES;RJ;SP;PR;SC;RS;MS;MT;GO;DF'.split(';')))
smp['UF'] = smp['Código IBGE Município'].transform(lambda z: estados[z // 1e5])
smp['Cidade'] = smp.apply(lambda z : f"{z['Município']}/{z['UF']}",axis=1)
# %%
def medmean(z: pd.DataFrame, col: str):
    """
    devolver informações de uma coluna do dataframe
    """
    res = pd.Series(data=[int(z[col].mean()), int(z[col].std()), z[col].median(), int(z[col].min()), int(z[col].max())], index=['Média','Desvio','Mediana','Mínimo','Máximo'],name=col)
    return res
caracteristicas = pd.DataFrame([medmean(smp, x) for x in colunas_analise])
for col in ['Média', 'Desvio',  'Mínimo', 'Máximo']:
    caracteristicas.loc[:,col]=caracteristicas.loc[:,col].astype(int)
print(caracteristicas.to_latex())

# %%
#Clusters Ward
def clusters_ward_por_nclusters(X, n_clusters=[2]):
    res = {}
    for n in n_clusters:
        res[n] = cluster.AgglomerativeClustering(n_clusters=n, ).fit_predict(X)
    return res
# %%
clusters_ward = clusters_ward_por_nclusters(dados_analise, [5, 8, 10])
smp['cl_ward'] = clusters_ward[8]
# %%
def descrever_grupo(z: pd.DataFrame, col: str):
    res = pd.Series(data=[
        z[col].count(), z[col].mean(), np.quantile(z[col], 0.25), z[col].median(), np.quantile(z[col], 0.75)
    ],
        index=['N° mun.', 'Média', 'Q1', 'Mediana', 'Q3'],
        name=col)
    return res
estats_clusters = smp.groupby(by=['cl_ward']).apply(descrever_grupo,'PF_Dados')
print(estats_clusters.to_latex())

# %%
#Alterar a forma de visualizar os dados
smp['acc_pop'] = smp.apply(lambda z : (z['PF_VOZ'] + z['PF_Dados'] + z['PF_VOZ_DADOS'])/z['pop'],axis=1)
smp['rel_voz_dados'] = smp['PF_VOZ_DADOS'] / smp['PJ_VOZ_DADOS']

# %%
fig, ax = plt.subplots(nrows=2, figsize=(16, 16))
ordem=[0,1,2,3,4,5,6,7]
sbn.scatterplot(data=smp, x='PF_VOZ_DADOS', y='PJ_VOZ_DADOS', hue='cl_ward', hue_order=ordem, legend='full', ax=ax[0], palette='tab10')
sbn.scatterplot(data=smp, x='acc_pop', y='rel_voz_dados', hue='cl_ward', hue_order=ordem, legend='full', ax=ax[1], palette='tab10')

# %%
analise_comparacao = ['pop', 'PF_Dados', 'PF_VOZ', 'PF_VOZ_DADOS', 'PJ_Dados', 'PJ_M2M_ESPECIAL', 'PJ_M2M_PADRAO', 'PJ_VOZ', 'PJ_VOZ_DADOS']
for col in [x for x in analise_comparacao if x != 'pop']:
    smp[f"{col}_dens"]=smp.apply(lambda z : z[col]/z['pop'],axis=1)
fig, ax = plt.subplots(nrows=len(analise_comparacao), figsize=(16, len(analise_comparacao) * 6))
for i, estat in enumerate(analise_comparacao):
    sbn.barplot(data=smp, x='cl_ward', y=estat, hue='cl_ward', ax=ax[i])
    ax[i].set_title(f"Analise {estat}")

# %%
## Gostei muito desses gráficos, podem ser úteis para descrever
analise_comparacao = [ 'PF_Dados', 'PF_VOZ', 'PF_VOZ_DADOS', 'PJ_Dados', 'PJ_M2M_ESPECIAL', 'PJ_M2M_PADRAO', 'PJ_VOZ', 'PJ_VOZ_DADOS']
for col in analise_comparacao:
    smp[f"{col}_dens"]=smp.apply(lambda z : z[col]/z['pop'],axis=1)
fig, ax = plt.subplots(nrows=len(analise_comparacao), figsize=(16, len(analise_comparacao) * 6))
for i, estat in enumerate(analise_comparacao):
    sbn.boxplot(data=smp, x='cl_ward', y=f"{estat}_dens", ax=ax[i],width=0.8)
    ax[i].set_title(f"Analise {estat}")

# %%
analise_comparacao = [ 'PF_Dados', 'PF_VOZ', 'PF_VOZ_DADOS', 'PJ_Dados', 'PJ_M2M_ESPECIAL', 'PJ_M2M_PADRAO', 'PJ_VOZ', 'PJ_VOZ_DADOS']
for col in analise_comparacao:
    smp[f"{col}_dens"]=smp.apply(lambda z : z[col]/z['pop'],axis=1)
fig, ax = plt.subplots(nrows=len(analise_comparacao), figsize=(16, len(analise_comparacao) * 6))
for i, estat in enumerate(analise_comparacao):
    sbn.boxplot(data=smp, x='cl_ward', y=f"{estat}_dens", ax=ax[i],width=0.8)
    ax[i].set_title(f"Analise {estat}")
# %%
clusters = ['cl_ward']
comparar_clusters_dens=pd.melt(smp,id_vars=['Código IBGE Município', 'LATITUDE', 'LONGITUDE']+clusters,value_name='Valor',value_vars=['PF_Dados_dens','PF_VOZ_dens', 'PF_VOZ_DADOS_dens', 'PJ_Dados_dens','PJ_M2M_ESPECIAL_dens', 'PJ_M2M_PADRAO_dens', 'PJ_VOZ_dens','PJ_VOZ_DADOS_dens'], var_name='Comparador')
comparar_clusters=pd.melt(smp,id_vars=['Código IBGE Município', 'LATITUDE', 'LONGITUDE', 'Cidade']+clusters,value_name='Valor',value_vars=[
       'PF_Dados',
       'PF_VOZ', 'PF_VOZ_DADOS', 'PJ_Dados',
       'PJ_M2M_ESPECIAL', 'PJ_M2M_PADRAO', 'PJ_VOZ',
       'PJ_VOZ_DADOS'], var_name='Comparador')


# %%
fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(26, 60), sharey=False)
for i in range(8):
    cl_slice = comparar_clusters[comparar_clusters['cl_ward'] == i]
    n_munic = len(smp[smp['cl_ward'] == i])
    sbn.boxplot(data=cl_slice, x='Comparador', y='Valor', ax=ax[i])
    ax[i].set_title(f"Cluster {i} - {n_munic} municípios")
# %%
fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(26, 60), sharey=False)
for i in range(8):
    cl_slice = comparar_clusters_dens[comparar_clusters_dens['cl_ward'] == i]
    n_munic = len(smp[smp['cl_ward'] == i])
    sbn.boxplot(data=cl_slice, x='Comparador', y='Valor', ax=ax[i])
    ax[i].set_title(f"Cluster {i} - {n_munic} municípios")

# %%
resumo_cluster = smp.groupby(by=['cl_ward']).agg(n_mun=('Ano', 'count'), pop_m=('pop', 'mean'), pop_t=('pop','sum'))
resumo_cluster['Perc.'] = resumo_cluster['pop_t'].transform(lambda z: z / resumo_cluster['pop_t'].sum() * 100)
resumo_cluster.loc[:, 'pop_m'] = resumo_cluster.loc[:, 'pop_m'].transform(lambda z: format_string("%d", z, grouping=True))
resumo_cluster.loc[:, 'Perc.'] = resumo_cluster.loc[:, 'Perc.'].transform(lambda z: format_string("%.2f %%", z))
resumo_cluster.drop(labels=['pop_t'],axis=1,inplace=True)
resumo_cluster.rename(columns={'cl_warg':'Grupo','n_mun':'Municípios','pop_m':'Média popul.'},inplace=True)
print(resumo_cluster.to_latex())

# %%
# Cluster n -> Variar n(cl)
cl=7
fig, ax = plt.subplots(figsize=(16,6))
cl_slice = comparar_clusters[comparar_clusters['cl_ward'] == cl]
n_munic = len(smp[smp['cl_ward'] == cl])
sbn.boxplot(data=cl_slice, x='Comparador', y='Valor', ax=ax)
ax.set_title(f"Grupo {cl}, {n_munic} municípios")
print(f"Municípios:\n{', '.join(cl_slice['Cidade'].unique())}")
fig.savefig(f'cluster_{cl}.png')

# %%
#Carregar mapa dos municípios bahianos
with sessao() as ses:
    resmap = ses.get('https://github.com/marcio-mutti/data_science_iesb/raw/main/Mapas/municipios_baianos.geojson.bz2')
    if resmap.status_code != 200:
        raise RuntimeError("Não consegui recuperar mapa dos municípios Bahianos")
    mun_bahia = jsonloads(decompress(resmap.content))
cores=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
for i in range(len(mun_bahia['features'])):
    mun = mun_bahia['features'][i]['properties']['ibge']
    cl = smp.loc[smp['Código IBGE Município'] == mun, 'cl_ward'].item()
    mun_bahia['features'][i]['properties']['cor']=cores[cl]
# %%
#Hora de mapear
def pintador(feat):
    return {'fillOpacity': 0.7, 'weight': 1, 'fillColor': feat['properties']['cor'], 'color': feat['properties']['cor']}

mapa_bahia=fl.Map(location=[-12.420,-41.461], zoom_start=6)
fl.GeoJson(data=mun_bahia, style_function=pintador).add_to(mapa_bahia)
mapa_bahia