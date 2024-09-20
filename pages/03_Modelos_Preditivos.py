import streamlit as st
import pages.models.clustering_ranking_h as ch
import pages.models.autogluon_pedra as apc
import pages.models.autogluon_bolsa as ab

st.title("Modelos de Machine Learning")



tab0, tab1, tab2, tab3 = st.tabs(["Apresentação","Clusterização Hierárquica", "Previsão da Pedra-Conceito", "Previsão para concessão de bolsa de estudos"])

with tab0:
    st.markdown("""
        ## Apresentação

        - Modelo de clusterização hierárquica para compreensão dos fatores que interferem na classificação geral do aluno
        - Modelo de Previsão da Pedra-Conceito
        - Modelo de Previsão para concessão de bolsa de estudos
        """)
with tab1:
    st.markdown("# Modelo de clusterização hierárquica")
    ch.main()
with tab2:
    st.markdown("# Modelo de Previsão da Pedra-Conceito")
    if st.button("Executar Previsão da Pedra-conceito"):
        apc.main()
with tab3:
    st.markdown("# Modelo de Previsão para concessão de bolsa de estudos")
    if st.button("Executar Previsão de Bolsista"):
        ab.main()
    


