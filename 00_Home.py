import streamlit as st 

st.set_page_config(
    page_title="FIAP-DATATHON-3DTAT",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/carlosvneves/fiap_techchallenge04/contribute",
        'About': "# Fiap Datathon - Passos Mágicos"
    }
)


st.markdown("<h1 style='text-align: center'; >Datathon - 3DTAT</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center'; >Dashboard e Modelo Preditivo para o PEDE da ONG Passos Mágicos</h2>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center'; >Autor: Carlos Eduardo Veras Neves <br> rm 353068 </h4>", unsafe_allow_html=True)

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.image('images/img_fiap.jpeg', caption='FIAP -Alura Pós-Tech - 3DTAT (setembro/2024)', use_column_width=True, width=30)
with col2:
                
    st.page_link("pages/01_Apresentação.py", label="Entendimento do problema de negócio", icon="📊")
    st.page_link("pages/02_Análise_Exploratória.py", label="Base de dados e Análise Exploratória", icon="🛠️🔍")
    st.page_link("pages/03_Modelos_Preditivos.py", label="Modelos preditivos", icon="📈")
    st.page_link("pages/04_Observações.py", label="Observações (ou _insights_)", icon="💡")


