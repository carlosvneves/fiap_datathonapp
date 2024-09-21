import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

@st.cache_data
def load_data():
    # import file from data folder
    data = pd.read_csv("data/df_pooled_ranking_clean.csv")
    #data = pd.read_csv("../../data/df_pooled_ranking_clean.csv")
    df_2022 = pd.read_csv("data/df_2022_preproc_select.csv")
    return data, df_2022

def preprocess_df_2022(df_2022):
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
    df_2022["diff_fase"] = df_2022["fase"] - df_2022["fase_ideal"]
    df_2022.drop(columns=["fase_ideal", "fase"], inplace=True)
    min_cg = df_2022["cg"].min()
    max_cg = df_2022["cg"].max()
    df_2022["cg_normalized_inverted"] = 1 - (df_2022["cg"] - min_cg) / (max_cg - min_cg)
    return df_2022

def evaluate_features_importance(df, target_col, categorical_features):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_features = X.drop(columns=categorical_features).columns.tolist()
    numerical_transformer = SimpleImputer(strategy="mean")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    importances = model.named_steps["regressor"].feature_importances_
    onehot_features = (
        model.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    all_features = numerical_features + list(onehot_features)
    importance_df = pd.DataFrame(
        {"Feature": all_features, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    #st.dataframe(importance_df)
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Importância das Features em relação à classificação geral do aluno",
        color="Importance",
        color_continuous_scale="Blues",

    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          height=600 + len(importance_df) * 8
)
    st.plotly_chart(fig)
    st.markdown(f"""
               Pela alta correlação entre INDE com a classificação geral do aluno, 
               é importante considerar como que outras variáveis podem influenciar. 
               **Parâmetros do ajuste Random-Forest:**
               **R² Score: {r2:.2f}**/
               **Mean Absolute Error:{mae:.2f}**.
               Como o _ida_ e as pedras são altemente correlacionadas entre si e com o _inde_, 
               isto se reflete no gráfico acima.
               """)
    

def compute_correlations(df, target_col):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove(target_col)
    correlations = []
    for col in numerical_cols:
        corr = df[target_col].corr(df[col])
        correlations.append(corr)
    corr_df = pd.DataFrame({"Variable": numerical_cols, "Correlation": correlations})
    return corr_df

def plot_dendrogram(corr_df):
    corr_array = np.array(corr_df["Correlation"])
    distances = pdist(corr_array.reshape(-1, 1), metric="cityblock")
    Z = linkage(distances, method="average")
    fig = ff.create_dendrogram(
        corr_array.reshape(-1, 1),
        orientation="bottom",
        labels=corr_df["Variable"].values,
        linkagefun=lambda x: linkage(x, "average", "cityblock"),
    )
    fig.update_layout(width=800, height=400)
    st.plotly_chart(fig)

def plot_elbow_method(corr_df):
    corr_array = np.array(corr_df["Correlation"]).reshape(-1, 1)
    K = range(1, 10)
    inertias = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(corr_array)
        inertias.append(kmeans.inertia_)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(K),
            y=inertias,
            mode="lines+markers",
            marker=dict(color="blue"),
            line=dict(dash="solid"),
            name="Inércia",
        )
    )
    fig.update_layout(
        title="Método do Cotovelo para K-Means",
        xaxis_title="Número de Clusters K",
        yaxis_title="Inércia",
        xaxis=dict(tickmode="linear"),
        template="plotly_white",
    )
    st.plotly_chart(fig)

def plot_silhouette_scores(corr_df):
    corr_array = np.array(corr_df["Correlation"]).reshape(-1, 1)
    K = range(2, 10)
    silhouette_scores = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(corr_array)
        score = silhouette_score(corr_array, labels)
        silhouette_scores.append(score)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(K),
            y=silhouette_scores,
            mode="lines+markers",
            marker=dict(color="green"),
            line=dict(dash="solid"),
            name="Coeficiente de Silhueta",
        )
    )
    fig.update_layout(
        title="Análise do Coeficiente de Silhueta para K-Means",
        xaxis_title="Número de Clusters K",
        yaxis_title="Coeficiente de Silhueta",
        xaxis=dict(tickmode="linear"),
        template="plotly_white",
    )
    st.plotly_chart(fig)



def plot_clusters(corr_df, n_clusters=4):
    corr_array = np.array(corr_df["Correlation"]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(corr_array)
    corr_df["Cluster"] = labels
    
    fig = px.scatter(
        corr_df,
        x="Variable",
        y="Correlation",
        color="Cluster",
        title="K-Means Clustering das Variáveis Independentes",
        color_continuous_scale="blues",
    )
    fig.update_layout(xaxis_tickangle=-90)
    fig.update_layout(width=800, height=600)

    st.plotly_chart(fig)

def main():
    data, df_2022 = load_data()
    df_2022 = preprocess_df_2022(df_2022)
    st.markdown("#### 1. Análise de importância das features via Random Forest")
    categorical_features = ["sexo", "indicado_bolsa", "pedra"]
    df_to_evaluate = df_2022.drop(columns=["nome", "cg", "inde"])
    evaluate_features_importance(
        df_to_evaluate, "cg_normalized_inverted", categorical_features
    )
    st.divider()
    st.markdown("#### 2. Escolha do número de clusters")
    df_for_corr = df_to_evaluate.copy()
    le = LabelEncoder()
    df_for_corr["sexo_encoded"] = le.fit_transform(df_for_corr["sexo"])
    df_for_corr["indicado_bolsa_encoded"] = le.fit_transform(
        df_for_corr["indicado_bolsa"]
    )
    df_for_corr = pd.get_dummies(df_for_corr, columns=["pedra"])
    df_for_corr.drop(columns=["sexo", "indicado_bolsa"], inplace=True)
    corr_df = compute_correlations(df_for_corr, "cg_normalized_inverted")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Método do Cotovelo")
        plot_elbow_method(corr_df)
    with col2:
        st.markdown("##### Coeficientes de Silhueta")
        plot_silhouette_scores(corr_df)
    st.markdown("**De acordo com os gráficos acima, _k = 4_ é o número ótimo de clusters para a nossa base de dados.**") 
    st.divider()
    st.markdown("#### 3. Dendrograma")
    plot_dendrogram(corr_df)
    st.markdown("""**De acordo com o dendrograma acima, estão em :green[verde as variáveis que interferem negativamente] na classificação geral do aluno
               e em :red[vermelho as que interferem positivamente].**""")
    with st.expander("Ver tabela de correlação"):
        st.dataframe(corr_df.round(2).sort_values(by="Correlation", ascending=True))
    st.divider()
    st.markdown("#### 4. Visualização de Clusterização")
    plot_clusters(corr_df, n_clusters=4)
    st.markdown("""
                As variáveis se organizam em 4 clusters, como pode ser visto no gráfico acima.
                
                - **cluster 0**: iaa, ian,ponto_virada_encoded, rec_sintese, diff_fase
                - **cluster 1**: idade, qtd_aval, anos_pm
                - **cluster 2**: ieg, ida, ipv, destaque_ieg_resultado_encoded,destaque_ida_resultado_encoded,destaque_ipv_resultado_encoded
                - **cluster 3**:ips,ipp, sexo_encoded,indicado_bolsa_encoded
                
                Nota-se que clusterização é capaz de capturar padrões de agrupamento que podem auxiliar na compreensão dos fatores que 
                levam um determinado estudante a ter uma classificação melhor ou pior. De uma forma geral, o modelo de clusterização
                mostra que os indicadores acadêmicos (iaa, rec_sintese  - síntese das recomendações das equipes de avaliação) e 
                psicossociais (ian, ponto_virada_encoded, diff_fase) são os mais relevantes. 
                
                """)
    with st.expander("Ver base de dados usada para treinar e testar o modelo"): 
        st.dataframe(data, hide_index=True)
        st.dataframe(df_2022, hide_index=True)
