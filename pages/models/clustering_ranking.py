# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import streamlit as st
# Opcional: realizar um teste ANOVA para verificar se a variação de cg entre os anos é significativa
from scipy.stats import f_oneway
import plotly.express as px





def perform_clustering():
    # %%
    # Load the dataset
    data = pd.read_csv(
        "../../data/df_pooled_ranking_clean.csv"
    )  # Replace with your dataset path

    df = preproc_data(data)
    
    features_eval(df, "cg")
    
    
    # %%
    cg_2021 = set_cg(data[data["ano"] == 2021][["inde", "cg"]])
    cg_2022 = set_cg(data[data["ano"] == 2022][["inde", "cg"]])

    # fill cg values in the original dataframe with the new cg values for the corresponding year
    data.loc[cg_2021.index, "cg"] = cg_2021["cg"]
    data.loc[cg_2022.index, "cg"] = cg_2022["cg"]

    #data.sort_values(by="cg", ascending=True).head()
    data.to_csv("../../data/df_pooled_ranking_clean_with_cg.csv")
    
    # %%
    verify_time_dependency(data)
    
    # %% [markdown]
    # ## Teste com os dados de 2022, uma vez que não há diferença substancial ao longo dos anos para a definição do `cg`

    # %%
    df_2022 = pd.read_csv("../../data/df_2022_preproc_select.csv")
    
    

    # %%
    df_2022 = df_2022[
        [
            "nome",
            "idade",
            "cg",
            "pedra",
            "inde",
            "iaa",
            "ieg",
            "ips",
            "ida",
            "ipv",
            "ian",
            "ipp",
            "qtd_aval",
            "indicado_bolsa",
            "sexo",
            "ponto_virada_encoded",
            "fase",
            "anos_pm",
            "fase_ideal",
            "destaque_ieg_resultado_encoded",
            "destaque_ida_resultado_encoded",
            "destaque_ipv_resultado_encoded",
            "rec_sintese",
        ]
    ].copy()


    df_2022.head()


    # %%
    df_2022["diff_fase"] = df_2022["fase"] - df_2022["fase_ideal"]
    df_2022.drop(columns=["fase_ideal", "fase"], inplace=True)
    df_2022.head()


    # %%
    df_2022.to_csv("../../data/df_2022_clustering.csv")
    
    # %% [markdown]
    # ### Adequação de `cg` para o algoritmo - normalização e inversão (quanto menor, melhor)

    # %%
    df = pd.read_csv("../../data/df_2022_clustering.csv")

    # Definir o valor mínimo e máximo da coluna "cg"
    min_cg = df["cg"].min()
    max_cg = df["cg"].max()

    # Aplicar a normalização inversa para refletir a ordem de melhor ranking
    df["cg_normalized_inverted"] = 1 - (df["cg"] - min_cg) / (max_cg - min_cg)
    
    # %%
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df.isna().sum()


    # %%
    corr_matrix = df.drop(
        columns=["nome", "pedra", "sexo", "indicado_bolsa"], axis=1
    ).corr()
    
    with st.expander("Mostrar Matriz de Correlação"):

        st.dataframe(corr_matrix)  # corr_matrix.style.background_gradient(cmap="coolwarm")


    # %%
    st.write("### Análise de Importância dos Features")

    # Selecionar a coluna alvo
    target_col = st.selectbox("Selecione a coluna alvo (variável dependente):", df.columns)

    # Selecionar as colunas categóricas
    all_columns = df.columns.tolist()
    all_columns.remove(target_col)
    categorical_features = st.multiselect(
        "Selecione as colunas categóricas:", options=all_columns
    )

    if st.button("Avaliar Importância das Features"):
        evaluate_features_importance(df, target_col, categorical_features)
    




def verify_time_dependency(data):
        
    # %% [markdown]
    # ## Faz sentido realizar a análise considerando a variação do `cg` no tempo?

    # %%
    # Carregar os dados
    df = data.copy().reset_index().rename({"nome": "aluno_id"}, axis=1)


    # 1. Agrupar os dados por aluno e ano, selecionando as colunas relevantes
    # Suponho que exista uma coluna 'aluno_id' que identifica cada aluno e 'ano' para o ano específico
    # Adapte o nome da coluna 'aluno_id' caso necessário
    #df_sorted = df.sort_values(by=["aluno_id", "ano"])

    # 2. Calcular a diferença do ranking cg entre os anos para cada aluno
    df["cg_diff_2021_2020"] = df.groupby("aluno_id")["cg"].diff(periods=1)
    df["cg_diff_2022_2021"] = df.groupby("aluno_id")["cg"].diff(periods=2)

    # 3. Visualizar a evolução do ranking de cada aluno ao longo do tempo com um gráfico de linhas
    st.title("Evolução do Ranking (cg) de Alunos ao Longo dos Anos")

    fig_line = px.line(
        df,
        x="ano",
        y="cg",
        color="aluno_id",
        markers=True,
        title="Evolução do Ranking (cg) de Alunos ao Longo dos Anos"
    )
    st.plotly_chart(fig_line)

    # 4. Criar boxplots para comparar a distribuição de cg em cada ano
    st.title("Distribuição do Ranking (cg) por Ano")

    fig_box = px.box(
        df,
        x="ano",
        y="cg",
        title="Distribuição do Ranking (cg) por Ano"
    )
    st.plotly_chart(fig_box)

    # 5. Analisar a variação do ranking de cada aluno
    st.title("Variação do Ranking de Cada Aluno")

    df_variation = df.groupby("aluno_id")["cg"].agg(["mean", "std"]).reset_index()
    st.write(df_variation)

    # Teste ANOVA para ver se há diferença significativa nos rankings entre os anos
    st.title("Análise de Variância (ANOVA) dos Rankings por Ano")

    # Criar listas com os valores de cg por ano
    anos = df["ano"].unique()
    cg_por_ano = [df[df["ano"] == ano]["cg"] for ano in anos]

    # Executar o teste ANOVA
    anova_result = f_oneway(*cg_por_ano)

    st.write(f"Resultado do teste ANOVA: F={anova_result.statistic:.2f}, p-value={anova_result.pvalue:.4f}")

    if anova_result.pvalue < 0.05:
        st.write("Há uma diferença estatisticamente significativa nos rankings entre os anos.")
    else:
        st.write("Não há diferença estatisticamente significativa nos rankings entre os anos.")


    # %% [markdown]
    # ### Resultado da ANOVA quanto à variação temporal da classificação no tempo:
    # 
    # O p-value de 0.556 é maior que o nível de significância usual de 0.05 (ou 5%). Isso significa que não há evidências suficientes para rejeitar a hipótese nula.
    # 
    # Hipótese nula (H₀): O ranking (cg) dos alunos não varia significativamente entre os anos.
    # Hipótese alternativa (H₁): O ranking (cg) dos alunos varia significativamente entre os anos.
    # Dado que o p-value é alto, não podemos concluir que existe uma diferença significativa entre os rankings nos diferentes anos. Em outras palavras, o teste ANOVA sugere que a variação dos rankings entre os anos de 2020, 2021 e 2022 não é estatisticamente significativa.
    # 
    # Logo, uma abordagem razoável é utilizar a média dos indicadores e nos outros casos somente os dados de 2022.

def plot_hcluster():
    pass 

def perform_kmeans():
    pass

def plot_kmeans():
    pass 

def preproc_data(data):
    # %% [markdown]
    # # Entendendo os fatores que levam a um melhor desempenho na classificação geral (cg)

    # %%
    data.set_index("nome", inplace=True)

    data["ano"] = data["ano"].astype(int)

    data["bolsista_encoded"] = data["bolsista_encoded"].astype(int)

    data["cg"] = data["cg"].astype(int)

    data["ian"] = data["ian"].astype("category")

    data["na_fase"] = data["na_fase"].astype(int)

    data["ponto_virada_encoded"] = data["ponto_virada_encoded"].astype(int)

    data["sexo_encoded"] = data["sexo_encoded"].astype(int)

    # %%
    data.loc[data["ano"] != 2022, "cg"] = np.nan


    # %% [markdown]
    # ## Considerando que não temos o ranking dos anos anteriores, qual a variável definitiva para a sua definição?

    # %%
    df = data.copy()
    df = df.dropna(subset=["cg"])
    
    return df


def features_eval(df, target_feature):
    # Step 1: Feature Selection using Random Forest
    features_for_importance = df.drop(columns=[target_feature])
    target_for_importance = df[target_feature]

    # Standardize the features
    scaler = StandardScaler()
    features_scaled_for_importance = scaler.fit_transform(features_for_importance)

    # Train a Random Forest to get feature importance
    rf = RandomForestRegressor(random_state=42)
    rf.fit(features_scaled_for_importance, target_for_importance)

    # Get feature importances and sort them
    importances = rf.feature_importances_
    feature_names = features_for_importance.columns
    sorted_indices = np.argsort(importances)[::-1]

    # Print the most important features
    print("Feature ranking:")
    for i in sorted_indices:
        print(f"{feature_names[i]}: {importances[i]}")
    return feature_names[sorted_indices]




# %% [markdown]
# A variável que dita a classificação geral é o `inde`. Então vamos definir a classificação para os anos anteriores:

# %%
def set_cg(df):
    df = df.sort_values(by="inde", ascending=False)
    df.reset_index(inplace=True)
    df["cg"] = pd.Series(range(1, len(df) + 1))
    df.set_index("nome", inplace=True)

    return df


# %%
# data.info()


# # %%
# df = data.reset_index().copy()
# df = df[df["ano"] == 2022]
# df = df.drop(columns=["ano"], axis=1)
# df.set_index("nome", inplace=True)
# df.head()






# Exibir as primeiras linhas da normalização invertida
# df_sorted_inverted = df[["cg", "cg_normalized_inverted"]].sort_values(by="cg").head()
# print(df_sorted_inverted)


# %% [markdown]
# ### Random Forest para análise de fatores que contribuem para melhor cg


# %% [markdown]
# ### Análise simplifica de clusterização

# %%
def evaluate_features_importance(df, target_col, categorical_features):
    # Definir as variáveis independentes e dependentes
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identificar as features numéricas
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [col for col in numerical_features if col not in categorical_features]

    # Pré-processamento para variáveis numéricas e categóricas
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Criação do ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Pipeline com pré-processamento e modelo
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"**R²:** {r2:.2f}")
    st.write(f"**MAE:** {mae:.2f}")

    # Obter importâncias das features
    importances = model.named_steps["regressor"].feature_importances_

    # Extrair nomes das features após a transformação
    onehot_feature_names = model.named_steps["preprocessor"].transformers_[1][1]\
        .named_steps["onehot"].get_feature_names_out(categorical_features)
    all_features = numerical_features + list(onehot_feature_names)

    # Criar DataFrame de importância
    importance_df = pd.DataFrame(
        {"Feature": all_features, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    st.write("### Importância das Features")
    st.dataframe(importance_df)

    # Plotar o gráfico de barras horizontal usando Plotly
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation='h',
        title="Importância das Features"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)

# Código principal do aplicativo Streamlit
st.title("Avaliação da Importância das Features")



# %%
# df_to_evaluate = df.drop(columns=["nome", "cg", "inde", "pedra", "ida"])

# evaluate_features_importance(
#     df_to_evaluate,
#     "cg_normalized_inverted",
#     ["sexo", "indicado_bolsa"],
# )


# # %%
# df_to_evaluate = df.drop(columns=["nome", "cg", "inde"])
# evaluate_features_importance(
#     df_to_evaluate, "cg_normalized_inverted", ["sexo", "indicado_bolsa", "pedra"]
# )


# # %%
# df_to_evaluate.columns


# %%
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder, StandardScaler

df_to_evaluate = df.drop(columns=["nome", "cg", "inde"])


# Pré-processamento de variáveis categóricas
# Codificar a variável 'Genero'
le = LabelEncoder()
df_to_evaluate["sexo_Encoded"] = le.fit_transform(df["sexo"])
df_to_evaluate["bolsa_Encoded"] = le.fit_transform(df["indicado_bolsa"])

# Aplicar One-Hot Encoding na variável 'Curso'
df_pedra_encoded = pd.get_dummies(df["pedra"], prefix="predra")

# Concatenar as variáveis codificadas ao DataFrame original
df_to_evaluate = pd.concat([df_to_evaluate, df_pedra_encoded], axis=1)

# Visualizar o DataFrame atualizado
print("DataFrame com variáveis categóricas codificadas:")
print(df_to_evaluate.head())


# %%
df_to_evaluate.columns


# %%
# Variáveis para normalização
variaveis_para_normalizar = [
    "idade",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipv",
    "ian",
    "ipp",
    "qtd_aval",
    "ponto_virada_encoded",
    "anos_pm",
    "destaque_ieg_resultado_encoded",
    "destaque_ida_resultado_encoded",
    "destaque_ipv_resultado_encoded",
    "rec_sintese",
    "diff_fase",
    "sexo_Encoded",
    "bolsa_Encoded",
    "predra_Ametista",
    "predra_Quartzo",
    "predra_Topázio",
    "predra_Ágata",
]

# Aplicar a padronização
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df_to_evaluate[variaveis_para_normalizar])
df_normalizado = pd.DataFrame(dados_normalizados, columns=variaveis_para_normalizar)

# Visualizar os dados normalizados
print("\nDados normalizados:")
print(df_normalizado.head())

# Lista de variáveis independentes atualizada
variaveis_independentes = variaveis_para_normalizar

# Calcular as correlações
correlacoes = []
for var in variaveis_independentes:
    corr = df_to_evaluate["cg_normalized_inverted"].corr(df_normalizado[var])
    correlacoes.append(corr)

# Criar um DataFrame com as correlações
df_correlacoes = pd.DataFrame(
    {"Variavel": variaveis_independentes, "Correlacao": correlacoes}
)

print("\nCorrelações entre as variáveis independentes e o desempenho acadêmico:")
print(df_correlacoes)


# %%
# Converter as correlações em um array numpy
correlacoes_array = np.array(correlacoes)

# Calcular a matriz de distância usando a diferença absoluta
distancias = pdist(correlacoes_array.reshape(-1, 1), metric="cityblock")

# Realizar a clusterização hierárquica
Z = linkage(distancias, method="average")

# Plotar o dendrograma
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=variaveis_independentes, leaf_rotation=90, leaf_font_size=10)
plt.title("Dendrograma das Variáveis Independentes")
plt.xlabel("Variáveis")
plt.ylabel("Distância")
plt.show()


# %%
import plotly.figure_factory as ff

# Converter as correlações em um array numpy
correlacoes_array = np.array(correlacoes)

# Calcular a matriz de distância usando a diferença absoluta
distancias = pdist(correlacoes_array.reshape(-1, 1), metric="cityblock")

# Realizar a clusterização hierárquica
Z = linkage(distancias, method="average")

# Criar o dendrograma usando Plotly
fig = ff.create_dendrogram(
    correlacoes_array.reshape(-1, 1),
    orientation="bottom",
    labels=variaveis_independentes,
    linkagefun=lambda x: linkage(x, method="average", metric="cityblock"),
)

# Ajustar o layout para evitar sobreposição das legendas
fig.update_layout(
    width=1200,
    height=600,
    title="Dendrograma das Variáveis Independentes",
    xaxis_title="Variáveis",
    yaxis_title="Distância",
    xaxis=dict(tickangle=-90, tickfont=dict(size=10), automargin=True),
    margin=dict(b=200),
)

# Exibir o dendrograma
fig.show()


# %%
df_correlacoes


# %%


# %%
from sklearn.mixture import GaussianMixture

n_components = np.arange(1, 10)
models = [
    GaussianMixture(n_components=n, random_state=42).fit(correlacoes_array)
    for n in n_components
]

plt.figure(figsize=(8, 5))
plt.plot(
    n_components, [m.bic(correlacoes_array) for m in models], label="BIC", marker="o"
)
plt.plot(
    n_components, [m.aic(correlacoes_array) for m in models], label="AIC", marker="o"
)
plt.xlabel("Número de componentes")
plt.ylabel("Critério de Informação")
plt.title("BIC e AIC para GMM")
plt.legend()
plt.show()


# %%
from sklearn.mixture import GaussianMixture

k = 3

# Aplicar GMM
gmm = GaussianMixture(n_components=k, random_state=42)
gmm.fit(correlacoes_array.reshape(-1, 1))
labels = gmm.predict(correlacoes_array.reshape(-1, 1))

labels


# %%
correlacoes_array


# %%
df_clusters = df_correlacoes.copy()
# Atualizar o DataFrame com os novos labels
df_clusters["Cluster"] = labels

df_clusters.head()


# %%
# Visualizar os clusters
plt.figure(figsize=(10, 6))
for cluster in range(k):
    cluster_data = df_clusters[df_clusters["Cluster"] == cluster]
    plt.scatter(
        cluster_data["Variavel"], cluster_data["Correlacao"], label=f"Cluster {cluster}"
    )
plt.title("GMM Clustering das Variáveis Independentes")
plt.xlabel("Variáveis")
plt.ylabel("Correlação")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Intervalo de valores para K
K = range(1, 10)
inertias = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(correlacoes_array)
    inertias.append(kmeans.inertia_)

# Plotar o gráfico do Método do Cotovelo
plt.figure(figsize=(8, 5))
plt.plot(K, inertias, "bo-")
plt.xlabel("Número de clusters K")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo para K-Means")
plt.show()


# %%
import plotly.graph_objects as go

# Método do Cotovelo usando Plotly
K = range(1, 10)
inertias = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(correlacoes_array)
    inertias.append(kmeans.inertia_)

# Criar o gráfico interativo com Plotly
fig_elbow = go.Figure()
fig_elbow.add_trace(
    go.Scatter(
        x=list(K),
        y=inertias,
        mode="lines+markers",
        marker=dict(color="blue"),
        line=dict(dash="solid"),
        name="Inércia",
    )
)

fig_elbow.update_layout(
    title="Método do Cotovelo para K-Means",
    xaxis_title="Número de Clusters K",
    yaxis_title="Inércia",
    xaxis=dict(tickmode="linear"),
    template="plotly_white",
)

fig_elbow.show()


# %%
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 10):  # O coeficiente de silhueta não está definido para k=1
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(correlacoes_array)
    score = silhouette_score(correlacoes_array, labels)
    silhouette_scores.append(score)

# Plotar o Coeficiente de Silhueta
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("Número de clusters K")
plt.ylabel("Coeficiente de Silhueta")
plt.title("Análise do Coeficiente de Silhueta para K-Means")
plt.show()


# %%
# Coeficiente de Silhueta usando Plotly
silhouette_scores = []
K_silhouette = range(2, 10)  # O coeficiente de silhueta não está definido para k=1

for k in K_silhouette:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(correlacoes_array)
    score = silhouette_score(correlacoes_array, labels)
    silhouette_scores.append(score)

# Criar o gráfico interativo com Plotly
fig_silhouette = go.Figure()
fig_silhouette.add_trace(
    go.Scatter(
        x=list(K_silhouette),
        y=silhouette_scores,
        mode="lines+markers",
        marker=dict(color="green"),
        line=dict(dash="solid"),
        name="Coeficiente de Silhueta",
    )
)

fig_silhouette.update_layout(
    title="Análise do Coeficiente de Silhueta para K-Means",
    xaxis_title="Número de Clusters K",
    yaxis_title="Coeficiente de Silhueta",
    xaxis=dict(tickmode="linear"),
    template="plotly_white",
)

fig_silhouette.show()


# %%
# Converter as correlações em um array numpy
correlacoes_array = np.array(correlacoes).reshape(-1, 1)

# Definir o número de clusters (por exemplo, 3)
k = 4

# Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(correlacoes_array)
labels = kmeans.labels_

# Criar um DataFrame com os resultados
df_clusters = pd.DataFrame(
    {"Variavel": variaveis_independentes, "Correlacao": correlacoes, "Cluster": labels}
)

# Visualizar os clusters
plt.figure(figsize=(10, 6))
for cluster in range(k):
    cluster_data = df_clusters[df_clusters["Cluster"] == cluster]
    plt.scatter(
        cluster_data["Variavel"], cluster_data["Correlacao"], label=f"Cluster {cluster}"
    )
plt.title("K-Means Clustering das Variáveis Independentes")
plt.xlabel("Variáveis")
plt.ylabel("Correlação")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# %%
import plotly.graph_objects as go

# Converter as correlações em um array numpy
correlacoes_array = np.array(correlacoes).reshape(-1, 1)

# Definir o número de clusters (por exemplo, 4)
k = 4

# Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(correlacoes_array)
labels = kmeans.labels_

# Criar um DataFrame com os resultados
df_clusters = pd.DataFrame(
    {"Variavel": variaveis_independentes, "Correlacao": correlacoes, "Cluster": labels}
)

# Ordenar o DataFrame para melhor visualização
df_clusters = df_clusters.sort_values(by="Correlacao").reset_index(drop=True)

# Visualizar os clusters usando Plotly
fig = go.Figure()

# Definir cores para os clusters
colors = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

for cluster in range(k):
    cluster_data = df_clusters[df_clusters["Cluster"] == cluster]
    fig.add_trace(
        go.Scatter(
            x=cluster_data["Variavel"],
            y=cluster_data["Correlacao"],
            mode="markers",
            marker=dict(size=10, color=colors[cluster % len(colors)]),
            name=f"Cluster {cluster}",
        )
    )

fig.update_layout(
    title="K-Means Clustering das Variáveis Independentes",
    xaxis_title="Variáveis",
    yaxis_title="Correlação",
    xaxis=dict(tickangle=-90),
    legend_title="Clusters",
    template="plotly_white",
)

fig.show()