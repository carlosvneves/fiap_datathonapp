import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import numpy as np


# Set the title
st.title("Análise Exploratória 🔍")

color_sequence = px.colors.sequential.Plasma

@st.cache_data
def load_data():
    df_2020_preproc = pd.read_csv("data/df_2020_preproc.csv")
    df_2021_preproc = pd.read_csv("data/df_2021_preproc.csv")
    df_2022_preproc = pd.read_csv("data/df_2022_preproc.csv")
    dicionario_de_dados = pd.read_csv("data/dicionario_dados.csv")
    df_pooled_common = pd.read_csv("data/df_pooled_common.csv").set_index("nome")
    return df_2020_preproc, df_2021_preproc, df_2022_preproc, dicionario_de_dados, df_pooled_common

# Basic descriptive statistics
def basic_descriptive_stats(df, column):
    return df[column].describe()

def proportions(df, group_by_column, count_column):
    return df.groupby([group_by_column, count_column]).size().reset_index(name='count')

# Plotly-based plots
def basic_plots(df):
    col1, col2 = st.columns(2)

    # Count by gender and pedra
    out = proportions(df, "sexo", "pedra")
    with col1:
        fig1 = px.bar(out, x='sexo', y='count', color='pedra', barmode='group',
                      labels={'sexo': 'Sexo', 'count': 'Contagem'},
                      title='Contagem normalizada por sexo e pedra',
                      color_discrete_sequence=color_sequence)
        st.plotly_chart(fig1, use_container_width=True)

    # KDE Distribution of Inde by Pedra
    with col2:
        fig2 = px.histogram(df, x='inde', color='pedra', nbins=50, histnorm='density',
                            marginal='rug', labels={'inde': 'Inde'},
                            title='Distribuição do Inde por Pedra',
                            color_discrete_sequence=color_sequence)
        st.plotly_chart(fig2, use_container_width=True)

    # Additional KDE plots for various comparisons
    plot_kde_by_feature(df, 'sexo', 'Inde por Sexo', col1)
    plot_kde_by_feature(df, 'bolsista_encoded', 'Inde por Bolsista', col2)
    plot_kde_by_feature(df, 'corraca', 'Inde por Cor/Raça', col1)

    # Boxplot of Inde by Pedra, Bolsista, Sexo
    boxplot_feature(df, 'pedra', 'Inde por Pedra', col2)
    boxplot_feature(df, 'bolsista_encoded', 'Inde por Bolsista', col1)
    boxplot_feature(df, 'sexo', 'Inde por Sexo', col2)

def plot_kde_by_feature(df, feature, title, col):
    fig = px.histogram(df, x='inde', color=feature, nbins=50, histnorm='density',
                       marginal='rug', labels={'inde': 'Inde'},
                       title=f'Distribuição do {title}',
                       color_discrete_sequence=color_sequence)
    col.plotly_chart(fig, use_container_width=True)

def boxplot_feature(df, feature, title, col):

    fig = px.box(df, x=feature, y='inde', color=feature,
                 labels={feature: title, 'inde': 'Inde'},
                 title=f'Boxplot do {title}',
                 color_discrete_sequence=color_sequence)
    col.plotly_chart(fig, use_container_width=True)

# Heatmap of correlation matrix
def corr_plot(df):
    corr_cols = ["anos_pm", "inde", "iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian", "bolsista_encoded", 
                 "ponto_virada_encoded", "pedra_encoded", "na_fase", "diff_fase", "idade", "sexo_encoded"]
    correlation_matrix = df[corr_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                    x=correlation_matrix.columns,
                                    y=correlation_matrix.index,
                                    colorscale='Blues', zmin=-1, zmax=1,
                                    colorbar_title="Correlação"))
    fig.update_layout(title='Heatmap da Matriz de Correlação', xaxis_nticks=36)
    st.plotly_chart(fig, use_container_width=True)

# Tab 0: Base de Dados
def tab_base_de_dados():
    st.header('Sobre a base de dados')
    st.markdown("""
                
    
        Foi fornecido aos alunos da Pós-Tech um conjunto de dados anonimizados sobre a PEDE dos anos de 2020, 2021 e 2022, sobre mais de 1.300 alunos pesquisados na PEDE. Esses dados por aluno são tais como:
        - Idade, sexo, cor/raça;
        - INDE e os indicadores que o compõe;
        - Anos na Passos Mágicos;
        - Fase e Turma;
        - Se o aluno é ou não bolsista;
        - Classificações do aluno no ranking geral, por turma e por fase;
        
        Os dados passaram por um processo de limpeza e tratamento, uma vez que, de acordo com o ano da pesquisa, foram coletados e preenchidos de modo diferente, refletindo certas mudanças metodológicas na pesquisa ao longo dos anos. Por exemplo, há diversas colunas que existem em um determinado ano, porém não existem em outro. Em casos como esse, quando possível, optou-se por realizar a imputação com base em outras colunas relacionadas.
        
        Em alguns casos outros tipos de tratamento foram aplicados aos dados, como no caso de colunas contendo dados textuais (com avaliação sobre os alunos), os quais tiveram que ser submetidos a um tratamento por meio de processamento de linguagem natural, de modo que pudessem se tornar informativos o suficiente. Em outros casos foi aplicada a estratégia de _encoding_ para utilização do dado dentro de um modelo.
        
        Quando necessário para a compreensão de determinado gráfico, o tratamento aplicado ao dado e o seu significado serão devidamente explicitados. 

    """)


# Tab 1: Estatísticas Descritivas
def tab_estatisticas_descritivas(df_2020_preproc, df_2021_preproc, df_2022_preproc, df_pooled_common, dicionario_de_dados):
    st.header('Estatísticas Descritivas por ano')
    ano = st.selectbox('Selecione o Ano', ['2020', '2021', '2022', 'Todos os anos'])
    
    df = df_pooled_common if ano == 'Todos os anos' else eval(f"df_{ano}_preproc")
    
    with st.expander('Ver Dados Completos'):
        st.dataframe(df)
    
    selected = st.radio("O que você gostaria de ver sobre os dados?", ["Dimensões", "Descrição dos campos", "Estatísticas descritivas", "Contagem de valores por campos"])
    coluna = st.selectbox('Selecione a coluna de interesse', df.columns)

    if selected == 'Estatísticas descritivas':
        if coluna:
            stats = basic_descriptive_stats(df, coluna)
            stats_df = stats.to_frame().reset_index().round(2).fillna('')
            stats_df.columns = ['Estatística', 'Valor']
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df.describe(include='all').round(2).fillna(''), use_container_width=True)

    elif selected == 'Contagem de valores por campos':
        if coluna:
            st.write('##### Contagem de valores por campos:')
            vc = df[coluna].value_counts().reset_index().rename(columns={'count':'Contagem'}).reset_index(drop=True)
            st.dataframe(vc, use_container_width=True, hide_index=True)
        else:
            vc = df.value_counts().reset_index().rename(columns={'count':'Contagem'}).reset_index(drop=True)
            st.dataframe(vc, use_container_width=True, hide_index=True)        
    elif selected == 'Descrição dos campos':
        st.write('##### Descrição dos campos:')
        if coluna:            
            resultado = dicionario_de_dados[dicionario_de_dados['Nome do Campo'].str.strip().str.lower() == coluna.strip().lower()]
            # Verificar se algum resultado foi encontrado
            if not resultado.empty:
                # Exibir as descrições encontradas
                for i, row in resultado.iterrows():
                    st.write(f"**Campo:** {row['Nome do Campo']}")                    
                    st.write(f"**Descrição:** {row['Descrição']}")
            else:
                st.write('Nenhum campo encontrado com essa chave de busca.')        
    else:
        if coluna:
            st.write('###### Os dados possuem a seguinte dimensão:', df[coluna].shape)
        else:
            st.write('###### Os dados possuem a seguinte dimensão:', df.shape)
   
    if st.button("Produzir Análise Exploratória Completa"): 
        with st.expander('Exibir'):
            pr = df.profile_report(minimal=True) # type: ignore
            if pr:
                st_profile_report(pr)
            else:
                st.write('Não disponível')      
            
            
            

# Tab 2: Visão Longitudinal
def tab_visao_longitudinal(df_pooled_common):
    st.header('Visão longitudinal')
    df_combined = df_pooled_common.copy()
    
    with st.expander('Ver Dados Longitudinais'):
        st.dataframe(df_combined)
    
    df_combined['inde'] = pd.to_numeric(df_combined['inde'], errors='coerce')
    col3, col4 = st.columns(2)
    
    # Evolution of INDE over time
    inde_por_ano = df_combined.groupby('ano')['inde'].mean().reset_index()
    with col3:
        fig1 = px.line(inde_por_ano, x='ano', y='inde', markers=True,
                       title='Evolução da Média do INDE (2020-2022)',
                       color_discrete_sequence=color_sequence)
        fig1.update_layout(xaxis_title='Ano', yaxis_title='Média do INDE', template='plotly_white')
        st.plotly_chart(fig1)

    inde_por_ano_sexo = df_combined.groupby(['ano', 'sexo'])['inde'].mean().reset_index()
    with col4:
        fig2 = px.line(inde_por_ano_sexo, x='ano', y='inde', color='sexo', markers=True,
                       title='Evolução da Média do INDE por Sexo (2020-2022)',
                       color_discrete_sequence=color_sequence)
        fig2.update_layout(xaxis_title='Ano', yaxis_title='Média do INDE', template='plotly_white')
        st.plotly_chart(fig2)

    # Stacked bar plot for Pedra classifications over time
    counts = df_combined.groupby(['ano', 'pedra']).size().reset_index(name='counts')
    fig3 = px.bar(counts, x='ano', y='counts', color='pedra', title='Distribuição das Classificações de Pedra (2020-2022)',
                  barmode='stack', color_discrete_sequence=color_sequence)
    fig3.update_layout(xaxis_title='Ano', yaxis_title='Número de Alunos', template='plotly_white')
    st.plotly_chart(fig3)

    # Evolution of percentage of students in each Pedra classification
    counts_total = df_combined.groupby(['ano']).size().reset_index(name='total')
    counts = counts.merge(counts_total, on='ano')
    counts['percent'] = (counts['counts'] / counts['total'] * 100).round(2)
    fig_new = px.line(counts, x='ano', y='percent', color='pedra', markers=True,
                      title='Evolução Percentual das Classificações de Pedra (2020-2022)',
                      color_discrete_sequence=color_sequence)
    fig_new.update_layout(xaxis_title='Ano', yaxis_title='Percentual de Alunos (%)', template='plotly_white')
    st.plotly_chart(fig_new)
    
    counts_total = df_combined.groupby(['ano']).size().reset_index(name='total')
    df_bolsistas = df_combined.groupby(['ano'])['bolsista_encoded'].sum().reset_index()
    df_bolsistas = df_bolsistas.merge(counts_total, on='ano')
    df_bolsistas['percent'] = (df_bolsistas['bolsista_encoded'] / df_bolsistas['total'] * 100).round(2)
    fig_new_1 = px.bar(df_bolsistas, x='ano', y='percent',
                      title='Evolução Percentual do percentual Bolsistas (2020-2022)',
                      color_discrete_sequence=color_sequence)
    fig_new_1.update_layout(xaxis_title='Ano', yaxis_title='Percentual de Alunos (%)', template='plotly_white')
    st.plotly_chart(fig_new_1)

    
    # Correlation of INDE with other variables
    corr_with_inde = df_combined.select_dtypes(include=['number'])
    corr_with_inde = corr_with_inde.corr()['inde'].drop('inde')

    corr_df = corr_with_inde.reset_index().rename(columns={'index': 'Variable', 'inde': 'Correlation'}).sort_values(by='Correlation', ascending=False)
    
    # Plot the correlation
    fig_new_2 = px.bar(corr_df, x='Correlation', y='Variable', orientation='h',
                 title='Correlação entre INDE e outras variáveis (2020-2022)',
                 color='Correlation', color_continuous_scale=color_sequence)
    fig_new_2.update_layout(xaxis_title='Correlação', yaxis_title='Variáveis')
    st.plotly_chart(fig_new_2)

 

    
    # Gráfico de histograma com Plotly Express
    fig4 = px.histogram(
        df_combined,
        x="inde",
        color="sexo",
        barmode="stack",
        title="Distribuição do INDE por sexo",
        template='plotly_white',        
    )

    # Cálculo dos quantis teóricos e dos dados para o Q-Q Plot
    qq = stats.probplot(df_combined["inde"], dist="norm", plot=None)
    quantis_teoricos = qq[0][0]
    quantis_dados = qq[0][1]
    slope, intercept, r = qq[1]

    # Gráfico Q-Q Plot com Plotly Graph Objects
    fig5 = go.Figure()

    # Adiciona os pontos dos quantis
    fig5.add_trace(go.Scatter(
        x=quantis_teoricos,
        y=quantis_dados,
        mode='markers',
        name='Dados'
    ))

    # Adiciona a linha de tendência
    x_line = np.array([quantis_teoricos.min(), quantis_teoricos.max()])
    y_line = intercept + slope * x_line # type: ignore
    fig5.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name='Linha de Tendência',
        
    ))

    fig5.update_layout(
        title="Q-Q Plot dos Dados",
        xaxis_title="Quantis Teóricos",
        yaxis_title="Quantis dos Dados",
        template='plotly_white',
        #color_sequence=color_sequence
        
    )

    # Exibe os gráficos lado a lado no Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig4)

    with col2:
        st.plotly_chart(fig5)

    
    st.markdown("---")
    st.write('#### Teste para verificar se há diferença estatisticamente significativa quanto ao INDE.')
    st.write('Como os dados não seguem uma distribuição normal, aplica-se os testes abaixo:')
    st.write('*Teste de Mann-Whitney*: **entre gêneros**')
    # Teste de Mann-Whitney
    male_inde = df_combined[df_combined['sexo']=='M']['inde']
    female_inde = df_combined[df_combined['sexo']=='F']['inde']

    mw_u,mw_p = stats.mannwhitneyu(male_inde, female_inde)
 
    st.write(f'u-estat:{mw_u:.2f}, p-valor:{mw_p:.2f}')
    st.markdown('Como :red[p-valor > 0.05], :blue-background[não há] diferença estatisticamente significativa')
    st.caption('Conclusão: o INDE evolui de modo similar entre alunos e alunas.')

    st.write('*Teste de Kruskal-Wallis*: **entre os anos**')
    # Teste de Kruskal-Wallis
    inde_2020 = df_combined[df_combined['ano']==2020]['inde']
    inde_2021 = df_combined[df_combined['ano']==2021]['inde']
    inde_2022 = df_combined[df_combined['ano']==2022]['inde']

    k_h, k_p =stats.kruskal(inde_2020, inde_2021, inde_2022)
    st.write(f'h-estat :{k_h:.2f}, p-valor :{k_p:.2f}')
    st.markdown('Como :red[p-valor < 0.05], :blue-background[há] diferença estatisticamente significativa.')
    st.caption('Conclusão: as variações do INDE entre os anos é significativa, sejam elas positivas ou negativas.')
    
    st.markdown("---")


# Tab 3: Visão Transversal
def tab_visao_transversal(df_2020_preproc, df_2021_preproc, df_2022_preproc):
    st.header('Visão Transversal')
    ano = st.selectbox('Selecione o Ano para os Gráficos', ['2020', '2021', '2022'], key='graficos')
    df = eval(f"df_{ano}_preproc")
    
    st.subheader('Heatmap de Correlação')  
    corr_plot(df)
    
    st.subheader('Gráficos para compreensão dos principais aspectos relacionados ao inde')  
    basic_plots(df)



df_2020_preproc, df_2021_preproc, df_2022_preproc, dicionario_de_dados, df_pooled_common = load_data()

aba0, aba1, aba2, aba3 = st.tabs(['Base de Dados', 'Estatísticas Descritivas', 'Visão longitudinal', 'Visão transversal'])

with aba0:
    tab_base_de_dados()

with aba1:
    tab_estatisticas_descritivas(df_2020_preproc, df_2021_preproc, df_2022_preproc, df_pooled_common, dicionario_de_dados)

with aba2:
    tab_visao_longitudinal(df_pooled_common)

with aba3:
    tab_visao_transversal(df_2020_preproc, df_2021_preproc, df_2022_preproc)



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.graph_objects as go
# from st_aggrid import AgGrid, GridOptionsBuilder
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# from streamlit_pandas_profiling import st_profile_report
# import ydata_profiling
# from scipy import stats


# st.title("Análise Exploratória")


# #@st.cache_data
# #def load_data():
# #################################################################################
# # Carregando os dataframes 
# #################################################################################
# df_2020_preproc = pd.read_csv("data/df_2020_preproc.csv")
# df_2021_preproc = pd.read_csv("data/df_2021_preproc.csv")
# df_2022_preproc = pd.read_csv("data/df_2022_preproc.csv")
# dicionario_de_dados = pd.read_csv("data/dicionario_dados.csv")

# #################################################################################
# #                                   Preparação do Modelo  Preditivo             #
# #################################################################################
# df_pooled_common = pd.read_csv("data/df_pooled_common.csv").set_index("nome")

# #################################################################################
# # Criação das abas
# aba0, aba1, aba2, aba3 = st.tabs(['Base de Dados','Estatísticas Descritivas', 'Visão longitudinal','Visão transversal'])
# # Funções existentes (mantenha as mesmas ou adapte conforme necessário)
# def basic_descriptive_stats(df, column):
#     return df[column].describe()
# def categorical_descriptive_stats(df, column):    
#     return df[column].value_counts()
# def proportions(df, group_by_column, count_column):
#     out = df.groupby([group_by_column, count_column]).size().reset_index(name='count')
#     return out
# def mann_witney_u_test(series1, series2):
#     return stats.mannwhitneyu(series1, series2)
# def kruskal_wallis_h_test(series1, series2, series3):
#     return stats.kruskal(series1, series2, series3)

# # Função para gráficos básicos usando Plotly
# def basic_plots(df):
#     out = proportions(df, "sexo", "pedra")
#     # Criação de colunas para dispor os gráficos em duas colunas
#     col1, col2 = st.columns(2)
#     # Sequência de cores com melhor contraste
#     color_sequence = ['#1f77b4', '#9467bd', '#17becf']
#     # Gráfico de Barras
#     with col1:
#         fig1 = px.bar(
#             out,
#             x='sexo',
#             y='count',
#             color='pedra',
#             barmode='group',
#             labels={'sexo': 'Sexo', 'count': 'Contagem'},
#             title='Contagem normalizada por sexo e pedra',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig1, use_container_width=True)
#     # Gráfico de Densidade (KDE) - Inde por Pedra
#     with col2:
#         fig2 = px.histogram(
#             df,
#             x='inde',
#             color='pedra',
#             nbins=50,
#             histnorm='density',
#             marginal='rug',
#             labels={'inde': 'Inde'},
#             title='Distribuição do inde por pedra',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig2, use_container_width=True)
#     # Gráfico de Densidade (KDE) - Inde por Sexo
#     with col1:
#         fig3 = px.histogram(
#             df,
#             x='inde',
#             color='sexo',
#             nbins=50,
#             histnorm='density',
#             marginal='rug',
#             labels={'inde': 'Inde'},
#             title='Distribuição do inde por sexo',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig3, use_container_width=True)
#     # Gráfico de Densidade (KDE) - Inde por Bolsista
#     with col2:
#         fig4 = px.histogram(
#             df,
#             x='inde',
#             color='bolsista_encoded',
#             nbins=50,
#             histnorm='density',
#             marginal='rug',
#             labels={'inde': 'Inde', 'bolsista_encoded': 'Bolsista'},
#             title='Distribuição do inde por bolsista',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig4, use_container_width=True)
#     # Gráfico de Densidade (KDE) - Inde por Cor/Raça
#     with col1:
#         fig5 = px.histogram(
#             df,
#             x='inde',
#             color='corraca',
#             nbins=50,
#             histnorm='density',
#             marginal='rug',
#             labels={'inde': 'Inde', 'corraca': 'Cor/Raça'},
#             title='Distribuição do inde por cor/raça',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig5, use_container_width=True)
#     # Boxplot - Inde por Pedra
#     with col2:
#         fig6 = px.box(
#             df,
#             x='pedra',
#             y='inde',
#             color='pedra',
#             labels={'pedra': 'Pedra', 'inde': 'Inde'},
#             title='Boxplot do inde por pedra',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig6, use_container_width=True)
#     # Boxplot - Inde por Bolsista
#     with col1:
#         fig7 = px.box(
#             df,
#             x='pedra',
#             y='inde',
#             color='bolsista_encoded',
#             labels={'pedra': 'Pedra', 'inde': 'Inde', 'bolsista_encoded': 'Bolsista'},
#             title='Boxplot do inde por bolsista',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig7, use_container_width=True)
#     # Boxplot - Inde por Sexo
#     with col2:
#         fig8 = px.box(
#             df,
#             x='pedra',
#             y='inde',
#             color='sexo',
#             labels={'pedra': 'Pedra', 'inde': 'Inde', 'sexo': 'Sexo'},
#             title='Boxplot do inde por sexo',
#             color_discrete_sequence=color_sequence
#         )
#         st.plotly_chart(fig8, use_container_width=True)
# # Função para plotar o heatmap de correlação
# def corr_plot(df):
#     # Sequência de cores com melhor contraste
#     color_sequence = ['#1f77b4', '#9467bd', '#17becf']
#     corr_cols = [
#         "anos_pm", "inde", "iaa", "ieg", "ips", "ida",
#         "ipp", "ipv", "ian", "bolsista_encoded",
#         "ponto_virada_encoded", "pedra_encoded", "na_fase",
#         "diff_fase", "idade", "sexo_encoded",
#     ]
#     destaque_cols = [
#         "destaque_ieg_resultado_encoded",
#         "destaque_ida_resultado_encoded",
#         "destaque_ipv_resultado_encoded",
#     ]
#     existing_destaque_cols = [col for col in destaque_cols if col in df.columns]
#     corr_cols.extend(existing_destaque_cols)
#     if "rec_sintese" in df.columns:
#         corr_cols.append("rec_sintese")
#     # Calculando a matriz de correlação
#     correlation_matrix = df[corr_cols].corr()
#     # Plotando a matriz de correlação como um heatmap
#     fig = go.Figure(data=go.Heatmap(
#         z=correlation_matrix.values,
#         x=correlation_matrix.columns,
#         y=correlation_matrix.index,
#         colorscale='blues',
#         zmin=-1,
#         zmax=1,
#         colorbar_title="Correlação",
#     ))
#     fig.update_layout(
#         title='Heatmap da Matriz de Correlação',
#         xaxis_nticks=36
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # def get_profile_report(ano):
# #     if ano == '2020':
# #         df = df_2020_preproc
# #     elif ano == '2021':
# #         df = df_2021_preproc
# #     else:
# #         df = df_2022_preproc
# #     
# #     return df.profile_report(minimal=True)
# #     



# # Conteúdo da Aba 0 - Base de Dados
# with aba0:
#     st.header('Sobre a base de dados')
#     st.markdown("""    
    
#     Foi fornecido aos alunos da Pós-Tech um conjunto de dados anonimizados sobre a PEDE dos anos de 2020, 2021 e 2022, sobre mais de 1.300 alunos pesquisados na PEDE. Esses dados por aluno são tais como:
#     - Idade, sexo, cor/raça;
#     - INDE e os indicadores que o compõe;
#     - Anos na Passos Mágicos;
#     - Fase e Turma;
#     - Se o aluno é ou não bolsista;
#     - Classificações do aluno no ranking geral, por turma e por fase;
    
#     Os dados passaram por um processo de limpeza e tratamento, uma vez que, de acordo com o ano da pesquisa, foram coletados e preenchidos de modo diferente, refletindo certas mudanças metodológicas na pesquisa ao longo dos anos. Por exemplo, há diversas colunas que existem em um determinado ano, porém não existem em outro. Em casos como esse, quando possível, optou-se por realizar a imputação com base em outras colunas relacionadas.
    
#     Em alguns casos outros tipos de tratamento foram aplicados aos dados, como no caso de colunas contendo dados textuais (com avaliação sobre os alunos), os quais tiveram que ser submetidos a um tratamento por meio de processamento de linguagem natural, de modo que pudessem se tornar informativos o suficiente. Em outros casos foi aplicada a estratégia de _encoding_ para utilização do dado dentro de um modelo.
    
#     Quando necessário para a compreensão de determinado gráfico, o tratamento aplicado ao dado e o seu significado serão devidamente explicitados. 
# """)




# # Conteúdo da Aba 1 - Estatísticas Descritivas
# with aba1:
#     st.header('Estatísticas Descritivas por ano')
#     # Selecionar o ano
#     ano = st.selectbox('Selecione o Ano', ['2020', '2021', '2022', 'Todos os anos'])
#     if ano == 'Todos os anos':
#         df = df_pooled_common 
#     elif ano == '2021':
#         df = df_2021_preproc
#     elif ano == '2022':
#         df = df_2022_preproc
#     else:
#         df = df_2020_preproc
        
    
#     # Mostrar dados completos com tabela interativa
#     #st.write('Dados Completos:')
#     with st.expander('Ver Dados Completos'):
#         # gb = GridOptionsBuilder.from_dataframe(df)
#         # gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
#         # gb.configure_default_column(editable=False, filter=True, sortable=True)
#         # gridOptions = gb.build()
#         # AgGrid(df, gridOptions=gridOptions, theme='balham')
#         st.dataframe(df)
    
#     if ano != 'Todos os anos':
#         # Selecionar a coluna
#         coluna = st.selectbox('Selecione a coluna de interesse', df.columns)
#     else:
#         coluna = None
    
    
#     selected = st.radio("O que você gostaria de ver sobre os dados?", 
#              ["Dimensões", 
#               "Descrição dos campos",
#               "Estatísticas descritivas",
#               "Contagem de valores por campos"])
    
#      #Showing summary statistics
#     if selected == 'Estatísticas descritivas':
#         st.write('#### Estatísticas Descritivas Básicas:')
        
#         if coluna:        
#             if df[coluna].dtype in ['float64', 'int64']:
#                 st.write(f'##### `{coluna}` é variável numérica')
#             else:
#                 st.write(f'##### `{coluna}` é variável categórica')

#             stats = basic_descriptive_stats(df, coluna)        
#             # Converter para DataFrame
#             stats_df = stats.to_frame().reset_index().round(2).fillna('')
#             stats_df.columns = ['Estatística', 'Valor']
#             #ss = pd.DataFrame(df[coluna].describe(include='all').round(2).fillna(''))
#             st.dataframe(stats_df, use_container_width=True, hide_index=True)
#         else:
#             ss = pd.DataFrame(df.describe(include='all').round(2).fillna(''))
#             st.dataframe(ss, use_container_width=True)
#     elif selected == 'Contagem de valores por campos':
#         if coluna:
#             st.write('##### Contagem de valores por campos:')
#             vc = df[coluna].value_counts().reset_index().rename(columns={'count':'Contagem'}).reset_index(drop=True)
#             st.dataframe(vc, use_container_width=True, hide_index=True)
#         else:
#             vc = df.value_counts().reset_index().rename(columns={'count':'Contagem'}).reset_index(drop=True)
#             st.dataframe(vc, use_container_width=True, hide_index=True)        
#     elif selected == 'Descrição dos campos':
#         st.write('##### Descrição dos campos:')
#         if coluna:            
#             resultado = dicionario_de_dados[dicionario_de_dados['Nome do Campo'].str.strip().str.lower() == coluna.strip().lower()]
#             # Verificar se algum resultado foi encontrado
#             if not resultado.empty:
#                 # Exibir as descrições encontradas
#                 for i, row in resultado.iterrows():
#                     st.write(f"**Campo:** {row['Nome do Campo']}")                    
#                     st.write(f"**Descrição:** {row['Descrição']}")
#             else:
#                 st.write('Nenhum campo encontrado com essa chave de busca.')
        
#     else:
#         if coluna:
#             st.write('###### Os dados possuem a seguinte dimensão:', df[coluna].shape)
#         else:
#             st.write('###### Os dados possuem a seguinte dimensão:', df.shape)
   
#     if st.button("Produzir Análise Exploratória Completa"): 
#         with st.expander('Exibir'):
#             pr = df.profile_report(minimal=True) # type: ignore
#             if pr:
#                 st_profile_report(pr)
#             else:
#                 st.write('Não disponível')
    

# with aba2:
#     st.header('Visão longitudinal')
    
#     # Unificando os três anos em um único dataframe
#     df_combined = df_pooled_common.copy()
#     with st.expander('Ver Dados Longitudinais'):
#         st.dataframe(df_combined)
    
#     col3, col4 = st.columns(2)
    


#     # # Convertendo o INDE para numérico
#     df_combined['inde'] = pd.to_numeric(df_combined['inde'], errors='coerce')

#     # # Gráfico 1: Evolução da média do INDE ao longo dos anos
#     inde_por_ano = df_combined.groupby('ano')['inde'].mean().reset_index()
    
#     with col3:
#         # # Criando o gráfico de linha para a evolução da média do INDE
#         fig1 = px.line(
#             inde_por_ano,
#             x='ano',
#             y='inde',
#             markers=True,
#             title='Evolução da Média do INDE (2020-2022)'
#         )
#         fig1.update_layout(
#             xaxis_title='Ano',
#             yaxis_title='Média do INDE',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig1)
    
#     with col4:
#         inde_por_ano_sexo = df_combined.groupby(['ano', 'sexo'])['inde'].mean().reset_index()

#         # # Criando o gráfico de linha para a evolução da média do INDE
#         fig2 = px.line(
#             inde_por_ano_sexo,
#             x='ano',
#             y='inde',
#             color='sexo',
#             markers=True,
#             title='Evolução da Média do INDE por sexo (2020-2022)'
#         )
#         fig2.update_layout(
#             xaxis_title='Ano',
#             yaxis_title='Média do INDE',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig2)

#     # # Gráfico 2: Distribuição das classificações de pedra por ano
#     counts = df_combined.groupby(['ano', 'pedra']).size().reset_index(name='counts') # type: ignore

#     # Criando o gráfico de barras empilhadas para a distribuição das classificações
#     fig3 = px.bar(
#         counts,
#         x='ano',
#         y='counts',
#         color='pedra',
#         title='Distribuição das Classificações de Pedra (2020-2022)',
#         barmode='stack'
#     )
#     fig3.update_layout(
#         xaxis_title='Ano',
#         yaxis_title='Número de Alunos',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig3)
    
#     # Gráfico de histograma com Plotly Express
#     fig4 = px.histogram(
#         df_combined,
#         x="inde",
#         color="sexo",
#         barmode="stack",
#         title="Distribuição do INDE por sexo"
#     )

#     # Cálculo dos quantis teóricos e dos dados para o Q-Q Plot
#     qq = stats.probplot(df_combined["inde"], dist="norm", plot=None)
#     quantis_teoricos = qq[0][0]
#     quantis_dados = qq[0][1]
#     slope, intercept, r = qq[1]

#     # Gráfico Q-Q Plot com Plotly Graph Objects
#     fig5 = go.Figure()

#     # Adiciona os pontos dos quantis
#     fig5.add_trace(go.Scatter(
#         x=quantis_teoricos,
#         y=quantis_dados,
#         mode='markers',
#         name='Dados'
#     ))

#     # Adiciona a linha de tendência
#     x_line = np.array([quantis_teoricos.min(), quantis_teoricos.max()])
#     y_line = intercept + slope * x_line # type: ignore
#     fig5.add_trace(go.Scatter(
#         x=x_line,
#         y=y_line,
#         mode='lines',
#         name='Linha de Tendência'
#     ))

#     fig5.update_layout(
#         title="Q-Q Plot dos Dados",
#         xaxis_title="Quantis Teóricos",
#         yaxis_title="Quantis dos Dados"
#     )

#     # Exibe os gráficos lado a lado no Streamlit
#     col1, col2 = st.columns(2)

#     with col1:
#         st.plotly_chart(fig4)

#     with col2:
#         st.plotly_chart(fig5)

    
#     st.markdown("---")
#     st.write('#### Teste para verificar se há diferença estatisticamente significativa quanto ao INDE.')
#     st.write('Como os dados não seguem uma distribuição normal, aplica-se os testes abaixo:')
#     st.write('*Teste de Mann-Whitney*: **entre gêneros**')
#     # Teste de Mann-Whitney
#     male_inde = df_combined[df_combined['sexo']=='M']['inde']
#     female_inde = df_combined[df_combined['sexo']=='F']['inde']

#     mw_u,mw_p = stats.mannwhitneyu(male_inde, female_inde)
 
#     st.write(f'u-estat:{mw_u:.2f}, p-valor:{mw_p:.2f}')
#     st.markdown('Como :red[p-valor > 0.05], :blue-background[não há] diferença estatisticamente significativa')
#     st.caption('Conclusão: o INDE evolui de modo similar entre alunos e alunas.')

#     st.write('*Teste de Kruskal-Wallis*: **entre os anos**')
#     # Teste de Kruskal-Wallis
#     inde_2020 = df_combined[df_combined['ano']==2020]['inde']
#     inde_2021 = df_combined[df_combined['ano']==2021]['inde']
#     inde_2022 = df_combined[df_combined['ano']==2022]['inde']

#     k_h, k_p =stats.kruskal(inde_2020, inde_2021, inde_2022)
#     st.write(f'h-estat :{k_h:.2f}, p-valor :{k_p:.2f}')
#     st.markdown('Como :red[p-valor < 0.05], :blue-background[há] diferença estatisticamente significativa.')
#     st.caption('Conclusão: as variações do INDE entre os anos é significativa, sejam elas positivas ou negativas.')
    
#     st.markdown("---")
    
    


# # Conteúdo da Aba 2 - Gráficos
# with aba3:
#     st.header('Visão Transversal')
#     # Selecionar o ano
#     ano = st.selectbox('Selecione o Ano para os Gráficos', ['2020', '2021', '2022'], key='graficos')
#     if ano == '2020':
#         df = df_2020_preproc
#     elif ano == '2021':
#         df = df_2021_preproc
#     else:
#         df = df_2022_preproc
    
#     st.subheader('Heatmap de Correlação')  
#     # Gerar e mostrar o heatmap de correlação
#     corr_plot(df)
    
#     st.subheader('Gráficos para compreensão dos principais aspectos relacionados ao inde')  
#     # Gerar e mostrar os gráficos
#     basic_plots(df)


