import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import numpy as np

#################################################################################
# Carregando os dataframes 
#################################################################################
df_2020_preproc = pd.read_csv("data/df_2020_preproc.csv")
df_2021_preproc = pd.read_csv("data/df_2021_preproc.csv")
df_2022_preproc = pd.read_csv("data/df_2022_preproc.csv")


#################################################################################
#                                   Preparação do Modelo  Preditivo             #
#################################################################################

df_pooled_common = pd.read_csv("data/df_pooled_common.csv").set_index("nome")

selected_columns = [
    "anos_pm",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipp",
    "ipv",
    "ian",
    "bolsista_encoded",
    "ponto_virada_encoded",
    "na_fase",
    "diff_fase",
    "idade",
    "sexo_encoded",
    "ano",
    "pedra",
]

df_model = df_pooled_common[selected_columns].copy().dropna()

# concatenate the two dataframes
# df_model = pd.concat(
#     [df_model, pd.get_dummies(df_model["ano"], dtype=int, prefix="d")], axis=1
# )

# # Remover 'ano' das features
# df_model = df_model.drop(columns=['ano'])

# Codificar a variável alvo 'pedra'
le_pedra = LabelEncoder()
df_model['pedra_encoded'] = le_pedra.fit_transform(df_model['pedra'])

df_model['diff_fase'] = df_model['diff_fase'].astype(int)

# Definir as features e a variável alvo
feature_columns = df_model.columns.drop(['pedra', 'pedra_encoded'])
X = df_model[feature_columns]
y = df_model['pedra_encoded']

# Dividir o conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Assuming y_test and y_pred are label encoded
le = LabelEncoder()
le.fit(
    df_pooled_common["pedra"].unique()
)  # original_class_names is a list of your class names
classes_ = le.classes_

# Acurácia
accuracy = accuracy_score(y_test, y_pred)

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Relatório de Classificação
class_report = classification_report(y_test, y_pred, target_names=classes_)

#################################################################################


# Configuração da página com cor de destaque turquesa
st.set_page_config(page_title='Dashboard Educacional', layout='wide')

primaryColor = "#17BECF"  # Azul Turquesa


# Título do Dashboard
st.title('Dashboard Educacional')

# Criação das abas
aba1, aba2, aba3, aba4, aba5 = st.tabs(['Estatísticas Descritivas', 'Gráficos', 'Heatmap de Correlação', 'Previsão de Pedra', 'Métricas do Modelo'])

# Funções existentes (mantenha as mesmas ou adapte conforme necessário)
def basic_descriptive_stats(df, column):
    return df[column].describe()

def categorical_descriptive_stats(df, column):
    return df[column].value_counts()

def proportions(df, group_by_column, count_column):
    out = df.groupby([group_by_column, count_column]).size().reset_index(name='count')
    return out

# Função para gráficos básicos usando Plotly
def basic_plots(df):
    out = proportions(df, "sexo", "pedra")

    # Criação de colunas para dispor os gráficos em duas colunas
    col1, col2 = st.columns(2)

    # Sequência de cores com melhor contraste
    color_sequence = ['#1f77b4', '#9467bd', '#17becf']

    # Gráfico de Barras
    with col1:
        fig1 = px.bar(
            out,
            x='sexo',
            y='count',
            color='pedra',
            barmode='group',
            labels={'sexo': 'Sexo', 'count': 'Contagem'},
            title='Contagem normalizada por sexo e pedra',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Gráfico de Densidade (KDE) - Inde por Pedra
    with col2:
        fig2 = px.histogram(
            df,
            x='inde',
            color='pedra',
            nbins=50,
            histnorm='density',
            marginal='rug',
            labels={'inde': 'Inde'},
            title='Distribuição do inde por pedra',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Gráfico de Densidade (KDE) - Inde por Sexo
    with col1:
        fig3 = px.histogram(
            df,
            x='inde',
            color='sexo',
            nbins=50,
            histnorm='density',
            marginal='rug',
            labels={'inde': 'Inde'},
            title='Distribuição do inde por sexo',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Gráfico de Densidade (KDE) - Inde por Bolsista
    with col2:
        fig4 = px.histogram(
            df,
            x='inde',
            color='bolsista_encoded',
            nbins=50,
            histnorm='density',
            marginal='rug',
            labels={'inde': 'Inde', 'bolsista_encoded': 'Bolsista'},
            title='Distribuição do inde por bolsista',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Gráfico de Densidade (KDE) - Inde por Cor/Raça
    with col1:
        fig5 = px.histogram(
            df,
            x='inde',
            color='corraca',
            nbins=50,
            histnorm='density',
            marginal='rug',
            labels={'inde': 'Inde', 'corraca': 'Cor/Raça'},
            title='Distribuição do inde por cor/raça',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Boxplot - Inde por Pedra
    with col2:
        fig6 = px.box(
            df,
            x='pedra',
            y='inde',
            color='pedra',
            labels={'pedra': 'Pedra', 'inde': 'Inde'},
            title='Boxplot do inde por pedra',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig6, use_container_width=True)

    # Boxplot - Inde por Bolsista
    with col1:
        fig7 = px.box(
            df,
            x='pedra',
            y='inde',
            color='bolsista_encoded',
            labels={'pedra': 'Pedra', 'inde': 'Inde', 'bolsista_encoded': 'Bolsista'},
            title='Boxplot do inde por bolsista',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig7, use_container_width=True)

    # Boxplot - Inde por Sexo
    with col2:
        fig8 = px.box(
            df,
            x='pedra',
            y='inde',
            color='sexo',
            labels={'pedra': 'Pedra', 'inde': 'Inde', 'sexo': 'Sexo'},
            title='Boxplot do inde por sexo',
            color_discrete_sequence=color_sequence
        )
        st.plotly_chart(fig8, use_container_width=True)


# Função para plotar o heatmap de correlação
def corr_plot(df):
    corr_cols = [
        "anos_pm", "inde", "iaa", "ieg", "ips", "ida",
        "ipp", "ipv", "ian", "bolsista_encoded",
        "ponto_virada_encoded", "pedra_encoded", "na_fase",
        "diff_fase", "idade", "sexo_encoded",
    ]

    destaque_cols = [
        "destaque_ieg_resultado_encoded",
        "destaque_ida_resultado_encoded",
        "destaque_ipv_resultado_encoded",
    ]

    existing_destaque_cols = [col for col in destaque_cols if col in df.columns]
    corr_cols.extend(existing_destaque_cols)

    if "rec_sintese" in df.columns:
        corr_cols.append("rec_sintese")

    # Calculando a matriz de correlação
    correlation_matrix = df[corr_cols].corr()

    # Plotando a matriz de correlação como um heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        colorbar_title="Correlação",
    ))

    fig.update_layout(
        title='Heatmap da Matriz de Correlação',
        xaxis_nticks=36
    )

    st.plotly_chart(fig, use_container_width=True)


# Conteúdo da Aba 1 - Estatísticas Descritivas
with aba1:
    st.header('Estatísticas Descritivas')

    # Selecionar o ano
    ano = st.selectbox('Selecione o Ano', ['2020', '2021', '2022'])
    if ano == '2020':
        df = df_2020_preproc
    elif ano == '2021':
        df = df_2021_preproc
    else:
        df = df_2022_preproc

    # Mostrar dados completos com tabela interativa
    st.write('Dados Completos:')
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
    gb.configure_default_column(editable=False, filter=True, sortable=True)
    gridOptions = gb.build()
    AgGrid(df, gridOptions=gridOptions, theme='balham')

    # Selecionar a coluna
    coluna = st.selectbox('Selecione a Coluna para mostrar a estatísticas descritiva', df.columns) 

    # Mostrar estatísticas descritivas
    if df[coluna].dtype in ['float64', 'int64']:
        stats = basic_descriptive_stats(df, coluna)
        st.write('Estatísticas Descritivas Básicas:')
        # Converter para DataFrame
        stats_df = stats.to_frame().reset_index()
        stats_df.columns = ['Estatística', 'Valor']
        # Configurar AgGrid
        gb = GridOptionsBuilder.from_dataframe(stats_df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        gb.configure_default_column(editable=False, filter=True, sortable=True)
        gridOptions = gb.build()
        AgGrid(stats_df, gridOptions=gridOptions, theme='balham')
    else:
        stats = categorical_descriptive_stats(df, coluna)
        st.write('Estatísticas Descritivas Categóricas:')
        # Converter para DataFrame
        stats_df = stats.reset_index()
        stats_df.columns = [coluna, 'Contagem']
        # Configurar AgGrid
        gb = GridOptionsBuilder.from_dataframe(stats_df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        gb.configure_default_column(editable=False, filter=True, sortable=True)
        gridOptions = gb.build()
        AgGrid(stats_df, gridOptions=gridOptions, theme='balham')

# Conteúdo da Aba 2 - Gráficos
with aba2:
    st.header('Gráficos')

    # Selecionar o ano
    ano = st.selectbox('Selecione o Ano para os Gráficos', ['2020', '2021', '2022'], key='graficos')
    if ano == '2020':
        df = df_2020_preproc
    elif ano == '2021':
        df = df_2021_preproc
    else:
        df = df_2022_preproc

    # Gerar e mostrar os gráficos
    basic_plots(df)

# Conteúdo da Aba 3 - Heatmap de Correlação
with aba3:
    st.header('Heatmap de Correlação')

    # Selecionar o ano
    ano = st.selectbox('Selecione o Ano para o Heatmap', ['2020', '2021', '2022'], key='heatmap')
    if ano == '2020':
        df = df_2020_preproc
    elif ano == '2021':
        df = df_2021_preproc
    else:
        df = df_2022_preproc

    # Gerar e mostrar o heatmap de correlação
    corr_plot(df)

# Conteúdo da Aba 4 - Previsão de Pedra
with aba4:
    st.header('Previsão da Pedra com Base nos Indicadores')

    user_input = {}

    # Variáveis Categóricas Codificadas
    encoded_categorical_cols = [
        'sexo_encoded', 'bolsista_encoded','na_fase', 'iaa', 'ponto_virada_encoded'
        
    ]

    for col in feature_columns:
        if col.startswith('d_'):
            continue  # As colunas dummies de 'ano' serão tratadas separadamente
        elif col in encoded_categorical_cols:
            # Mapear os valores codificados para opções legíveis
            if col == 'sexo_encoded':
                options = {'Feminino': 0, 'Masculino': 1}
                selected = st.selectbox('Sexo', options.keys())
                user_input[col] = options[selected]
            elif col == 'bolsista_encoded':
                options = {'Sim': 1, 'Não': 1}
                selected = st.selectbox('Bolsista', options.keys())
                user_input[col] = options[selected]
            elif col == 'na_fase':
                options = {'Sim': 1, 'Não': 0}  
                selected = st.selectbox('Na fase', options.keys())
                user_input[col] = options[selected]
            elif col == 'iaa':
                options = {'2.5': 2.5, '5.0': 5.0, '10.0': 10.0}
                selected = st.selectbox('iaa', options.keys())
                user_input[col] = options[selected]
            elif col == 'ponto_virada_encoded':
                options = {'Atingiu': 1, 'Não atingiu': 0}
                selected = st.selectbox('Ponto de virada', options.keys())
                user_input[col] = options[selected]
            else:
                # Para as variáveis de destaque
                options = {'Não Destacado': 0, 'Destacado': 1}
                label = col.replace('_encoded', '').replace('_', ' ').title()
                selected = st.selectbox(label, options.keys())
                user_input[col] = options[selected]
        elif col == 'ano':
            # Permitir que o usuário insira qualquer ano
            min_year = int(df_model['ano'].min())
            max_year = int(df_model['ano'].max()) + 5  # Permitir até 5 anos no futuro
            default_year = int(df_model['ano'].mean())
            user_input[col] = st.number_input(
                'Ano', min_value=min_year, max_value=max_year, value=default_year
            )
        elif col == 'diff_fase':
            # Usar slider para 'diff_fase' com valores inteiros de -7 a 7
            user_input[col] = st.slider(
                'diff_fase',
                min_value=-7,
                max_value=7,
                value=0,
                step=1
            )
        elif col == 'anos_pm':
            # Usar slider para 'anos_pm' com valores inteiros de 0 a 10
            user_input[col] = st.slider(
                'anos_pm',
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )
        else:
            # Variáveis Numéricas
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            user_input[col] = st.slider(f'{col}', min_val, max_val, mean_val)
    
    if st.button('Prever Pedra'):
        # Criar DataFrame com as entradas do usuário
        input_df = pd.DataFrame([user_input])

        # Garantir que as colunas estejam na ordem correta
        input_df = input_df[feature_columns]

        # Prever a pedra
        prediction = model.predict(input_df)
        predicted_pedra = le_pedra.inverse_transform(prediction)
        st.success(f'**A Pedra Prevista é:** {predicted_pedra[0]}')

# Conteúdo da Aba 5 - Métricas do Modelo
with aba5:
    st.header('Métricas de Desempenho do Modelo')
    st.write(f'**Acurácia do Modelo:** {accuracy:.2%}')
    st.subheader('Relatório de Classificação')
    st.text(class_report)
    st.subheader('Matriz de Confusão')
    fig_conf_matrix = px.imshow(
        conf_matrix,
        labels=dict(x="Predito", y="Real", color="Quantidade"),
        x=le_pedra.classes_,
        y=le_pedra.classes_,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_conf_matrix)
