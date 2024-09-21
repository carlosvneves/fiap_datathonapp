import streamlit as st
import pages.models.clustering_ranking_h as ranking_clustering
import pages.models.autogluon_pedra as pedra_classification
import pages.models.autogluon_bolsa as bolsa_classification
import pages.models.autogluon_pedra_predict as pedra_predict
import pages.models.autogluon_bolsa_predict as bolsa_predict

st.title("Modelos de Machine Learning 🧠")


# adicionar upload de arquivo de dados
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Apresentação",
        "Classificação Geral",
        "Pedra-Conceito",
        "Bolsa de Estudos",
        "Previsão da Pedra-Conceito",
        "Previsão de Bolsa de Estudos",
    ]
)

with tab0:
    st.markdown(
        """
        ### Entendendo os dados e realizando previsões por meio de Modelos de _Machine Learning_ disponíveis:

        A partir da leitura dos relatórios da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) e da análise exploratória de dados,
        foi identificado que as principais métricas para avaliar o desempenho dos alunos na Passos Mágicos são:

        - **Classificação Geral**
        - **Pedra-Conceito**
        - **Concessão de Bolsa de Estudos**

        Os modelos de _Machine-Learning_ são úteis não só para compreender como todos os indicadores disponíveis, bem como outros dados,
        são importantes ou não para definí-las, mas também para realizar previsões. Os modelos foram treinados e testados em dados de 2020, 2021 e 2022,
        os quais foram pré-processados para que possam ser utilizados como dados de treinamento e de teste.

        Para a **Classificação Geral** foi desenvolvido um modelo de clusterização hieráquica com Random Forest, de modo que é possível
        perceber a estrutura subjacente aos dados e identificar os padrões de agrupamento das diversas variáveis. Então se pode entender
        como os fatores que influenciam na classificação geral do aluno na Passos Mágicos.

        Para a **Pedra-Conceito** foi desenvolvido um **modelo de classificação do tipo _WeightedEnsemble_L3_** com o auxílio da biblioteca
        *Autogluon*. Este modelo apresentou resultados satisfatórios aoclassificar os alunos de acordo com sua Pedra-Conceito. Foi
        desenvolvida também uma interface do usuário, onde ele pode realizar simulações e previsões em função dos fatores mais relevantes,
        para um horizonte de até três anos.

        Um modelo de _Machine Learning_ também foi desenvolvido para permitir melhor compreensão e previsão da **Concessão de Bolsa de
        Estudos**. O treinamento e teste dos modelos foi realizado por meio da biblioteca *Autogluon* a qual apontou para
        um **modelo de classificação do tipo _WeightedEnsemble_L1_** como o mais adequado para a classificação dos
        alunos como bolsista ou não bolsista. Para previsão da concessão de bolsa de estudos, foi desenvolvida uma interface do usuário,
        onde ele pode realizar simulações e previsões em função dos fatores mais relevantes.

        Portanto, são cinco as abas disponíveis para a análise e compreensão dos dados por meio de _Machine Learning_:
        - **Classificação Geral**: modelo de clusterização para compreender os principais fatores que interferem na classificação geral dos alunos;
        - **Pedra-Conceito**: modelo de classificação para compreender a Pedra-Conceito dos alunos. É também o modelo base para a previsão da
        Pedra-Conceito;
        - **Concessão de Bolsa de Estudos**: modelo de classificação para compreender a concessão de bolsa de estudos dos alunos. É também o
        modelo base para a previsão da concessão de bolsa de estudos;
        - **Previsão da Pedra-Conceito**: modelo de previsão da Pedra-Conceito dos alunos;
        - **Previsão de Bolsa de Estudos**: modelo de previsão da concessão de bolsa de estudos dos alunos.
        """
    )

with tab1:
    st.markdown("# Quais fatores mais influenciam na classificação geral do aluno na Passos Mágicos?")
    ranking_clustering.main()
with tab2:
    st.markdown("# Modelo para classificação da Pedra-Conceito")
    pedra_classification.main()

with tab3:
    st.markdown("# Modelo para classificação dos alunos como aptos ou não para recebimento de bolsa de estudos")
    bolsa_classification.main()

with tab4:
    st.markdown("# Previsão da Pedra-Conceito")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            pedra_predict.main()
        with col2:
            st.image("images/pedras.png", use_column_width=True)

with tab5:
    st.markdown("#  Previsão da concessão de bolsa de estudos")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            bolsa_predict.main()
        with col2:
            st.image("images/scolarship.png", use_column_width=True)
