import time
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import os

@st.cache_data
def load_data():
    data = pd.read_csv("data/df_pooled_common.csv")
    return data

def preprocess_data(data):
    # Convert columns to appropriate data types
    data["ian"] = data["ian"].astype("category")
    data["sexo"] = data["sexo"].astype("category")
    data["pedra"] = data["pedra"].astype("category")
    data["ponto_virada"] = data["ponto_virada"].astype("category")
    data["fase"] = data["fase"].astype("category")
    data["na_fase"] = data["na_fase"].astype(bool)
    data["bolsista_encoded"] = data["bolsista_encoded"].astype(bool)
    data["ano"] = data["ano"].apply(lambda x: "t0" if x == 2020 else ("t1" if x == 2021 else "t2"))
    data["ano"] = data["ano"].astype("category")
    # Drop unnecessary columns
    data = data.drop(columns=["pedra_encoded", "nome", "corraca", "sexo_encoded"])
    return data

def prepare_datasets(data):
    X = data.drop(columns=["pedra", "inde"])
    X = X[["ida", "ieg", "ipv", "ano", "iaa", "ian", "ipp", "ips", "na_fase"]]
    y = data["pedra"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=41, shuffle=True
    )
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    return train_data, test_data, X_test

def train_or_load_model(train_data, label, save_path):
    
    
    if os.path.exists(save_path) and os.listdir(save_path):
        try:
            st.info('Carregando modelo salvo...')
            predictor = TabularPredictor.load(save_path)
        except Exception as e:
            st.warning(f'Falha ao carregar o modelo salvo: {e}')
            st.info('Treinando novo modelo...')
            predictor = TabularPredictor(
                label=label, path=save_path, problem_type="multiclass"
            ).fit(train_data, presets="good_quality", num_gpus=1)
    else:
        st.info('Treinando novo modelo...')
        predictor = TabularPredictor(
            label=label, path=save_path, problem_type="multiclass"
        ).fit(train_data, presets="good_quality", num_gpus=1)
    return predictor

def predict_with_user_input(predictor):
    st.markdown("### Por favor, insira os valores para previsão")
    

    # Define categorical and numeric fields
    anos = ["Ano presente", "Próximo ano", "Daqui a 2 anos"]
    ians = ["2.5", "5.0","10,0"]  
    na_fase_options = ["Sim", "Não"]
    
    # Create input fields
    ida = st.slider("Indicador de Desempenho Acadêmico (IDA)", min_value=0.0, step=0.1, max_value=10.0, value=5.0)
    ieg = st.slider("Indicador de Engajamento (IEG)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    ipv = st.slider("Indicador do Ponto de Virada (IPV)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    iaa = st.slider("Indicador de Autoavaliação (IAA)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    ipp = st.slider("Indicador Psicopedagógico (IPP)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    ips = st.slider("Indicador Psicossocial (IPS)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    ian = st.radio("Indicador de Adequação de Nível (IAN)", ians)
    na_fase = st.radio("Está na fase correta?", na_fase_options)
    ano = st.selectbox("Qual o horizonte da previsão (em anos)?", anos)
        
    ian = float(ian)
    
    if na_fase == "Sim":
        na_fase = True
    else:
        na_fase = False
    
    if ano == "Ano presente":
        ano = "t0"
    elif ano == "Próximo ano":
        ano = "t1"
    else:
        ano = "t2"
    
    # Create a dictionary of inputs
    user_input = {
        "ida": ida,
        "ieg": ieg,
        "ipv": ipv,
        "ano": ano,
        "iaa": iaa,
        "ian": ian,
        "ipp": ipp,
        "ips": ips,
        "na_fase": na_fase,
    }
    
    # Convert to dataframe
    input_df = pd.DataFrame([user_input])
  

    # Use model for prediction
    if st.button("Fazer Previsão"):
        prediction = predictor.predict(input_df)
        pedra = prediction.iloc[0]
        st.success(f"O resultado previsto para a 'pedra' é: {prediction.iloc[0]}")
        if pedra == "Quartzo":
            st.toast(f"#### O resultado previsto para a 'pedra' é: :red[{pedra}]")
            st.image("images/quartzo.png", use_column_width=True)
        if pedra == "Ágata":
            st.toast(f"#### O resultado previsto para a 'pedra' é: :orange[{pedra}]")
            st.image("images/agata.png", use_column_width=True)
        if pedra == "Ametista":
            st.toast(f"#### O resultado previsto para a 'pedra' é: :violet[{pedra}]")
            st.image("images/ametista.png", use_column_width=True)
        if pedra == "Topázio":
            st.toast(f"#### O resultado previsto para a 'pedra' é: :blue[{pedra}]")
            st.image("images/topazio.png", use_column_width=True)
        
        
def main():
    data = load_data()
    data = preprocess_data(data)
    train_data, test_data, X_test = prepare_datasets(data)
    label = "pedra"
    save_path = "data/agModels-predictPedra"
    
    with st.sidebar:
        with st.status("Treinando o Modelo ou Carregando Modelo Salvo - Pedra"):
            with st.spinner('Processando...'):
                predictor = train_or_load_model(train_data, label, save_path)
            st.success('Modelo pronto!')
        
    st.sidebar.write("")

    predict_with_user_input(predictor)
    
    
    