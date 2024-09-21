import streamlit as st
import pages.models.clustering_ranking_h as ranking_clustering
import pages.models.autogluon_pedra as pedra_classification
import pages.models.autogluon_bolsa as bolsa_classification
import pages.models.autogluon_pedra_predict as pedra_predict
import pages.models.autogluon_bolsa_predict as bolsa_predict

st.title("Modelos de Machine Learning")


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
        são importantes ou não para definí-las, mas também para realizar previsões. Os modelos foram treinados em dados de 2020, 2021 e 2022,
        os quais foram pré-processados para que possam ser utilizados como dados de treinamento e de teste.
              
        Para a **Classificação Geral** foi desenvolvido um modelo de clusterização hieráquica com Random Forest, de modo que é possível
        perceber a estrutura subjacente dos dados e identificar padrões de agrupamento das diversas variáveis. Então se pode entender 
        como os fatores que influenciam na classificação geral do aluno na Passos Mágicos.
        
        Para a **Pedra-Conceito** se desenvolveu um modelo de classificação do tipo _WeightedEnsemble_L3_ com o auxílio da biblioteca *Autogluon*.
        O modelo Para
        
        o modelo de _Machine Learning_ para a métrica de Concessão de Bolsa de Estudos  - **Classificação
        foi desenvolvido utilizando o algoritmo de regressão linear, e o modelo de _Machine Learning_ para a métrica de Classificação Geral 
        foi desenvolvido utilizando o algoritmo de classificação com Random Forest.
        
        
        dos alunos   
        - Modelo de _Machine-Learning_ para compreensão dos principais fatores que interferem na **Classificação Geral** de alunos
        - Modelo de _Machine-Learning_ para compreensão e previsão da **Pedra-Conceito** 
        - Modelo de _Machine-Learning_ para compreensão e previsão da **Concessão de Bolsa de Estudos**
        
        """
    )

with tab1:
    st.markdown("# Quais fatores mais influenciam na classificação geral do aluno na Passos Mágicos?")
    ranking_clustering.main()
with tab2:
    st.markdown("# Modelo para classificação da Pedra-Conceito")
    pedra_classification.main()
    
with tab3:
    st.markdown("# Modelo para classificação dos alunos para concessão de bolsa de estudos") 
    # if st.button(
    #     "Mostrar/Treinar Modelo para concessão de bolsa de estudos"
    # ):
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
    
