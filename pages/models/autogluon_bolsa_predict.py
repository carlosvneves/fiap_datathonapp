import time
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics

@st.cache_data
def load_data():
    data = pd.read_csv("data/df_pooled_common.csv")
    return data

def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(columns=["pedra_encoded", "nome"])
    # Transform 'bolsista_encoded' to 'bolsista' category 'sim'/'não'
    data["bolsista"] = data["bolsista_encoded"].apply(lambda x: "sim" if x == 1 else "não").astype("category")
    data = data.drop(columns=["bolsista_encoded"])
    # Transform 'pedra' to category
    data["pedra"] = data["pedra"].astype("category")
    # Map 'ano' to 't0', 't1', 't2'
    data["ano"] = data["ano"].apply(lambda x: "t0" if x == 2020 else ("t1" if x == 2021 else "t2"))
    data["ano"] = data["ano"].astype("category")
    return data

def prepare_datasets(data):
    # Select predictor variables
    X = data.drop(columns=["inde", "bolsista"])
    X = X[["anos_pm", "fase", "ida", "ipv", "ieg", "idade", "ipp", "na_fase", "sexo"]]
    y = data["bolsista"]
    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=41, shuffle=True
    )
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    return train_data, test_data, X_test

def train_or_load_model(train_data, label, save_path):
    # Check if the model exists and load it; otherwise, train a new model
    if os.path.exists(save_path) and os.listdir(save_path):
        try:
            st.info('Carregando modelo salvo...')
            predictor = TabularPredictor.load(save_path)
        except Exception as e:
            st.warning(f'Falha ao carregar o modelo salvo: {e}')
            st.info('Treinando novo modelo...')
            predictor = TabularPredictor(label=label, path=save_path).fit(
                train_data, presets="good_quality", num_gpus=1
            )
    else:
        st.info('Treinando novo modelo...')
        predictor = TabularPredictor(label=label, path=save_path).fit(
            train_data, presets="good_quality", num_gpus=1
        )
    return predictor

def predict_with_user_input(predictor):
    st.markdown("### Por favor, insira os valores para previsão")

    # Define categorical and numeric fields
    fases = ["fase 0", "fase 1", "fase 2", "fase 3", "fase 4", "fase 5", "fase 6", "fase 7"]  # Example phases
    sexos = ["M", "F"] 
    na_fase_options = ["Sim", "Não"]
    
    # Create input fields
    anos_pm = st.slider(".Anos na PM", min_value=0, max_value=10, value=5)
    idade = st.slider(".Idade-", min_value=10, max_value=20, value=15)
    ida = st.slider(".Indicador de Desempenho Acadêmico (IDA)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    ipv = st.slider(".Indicador do Ponto de Virada (IPV)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    ieg = st.slider(".Indicador de Engajamento (IEG)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    ipp = st.slider(".Indicador Psicopedagógico (IPP)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    fase = st.selectbox(".Fase na PM", fases)
    sexo = st.selectbox(".Sexo", sexos)
    na_fase = st.radio(".Está na fase?", na_fase_options)
    # converts the string to boolean
    if na_fase == "Sim":
        na_fase = True
    else:
        na_fase = False
    
    # converts the string to numeric
    fase = fase.replace("fase ", "")
    

    # Create a dictionary of inputs
    user_input = {
        "anos_pm": anos_pm,
        "fase": fase,
        "ida": ida,
        "ipv": ipv,
        "ieg": ieg,
        "idade": idade,
        "ipp": ipp,
        "na_fase": na_fase,
        "sexo": sexo
    }
    
    # Convert to dataframe
    input_df = pd.DataFrame([user_input])

    # Use model for prediction
    if st.button("Fazer Previsão da Bolsa"):
        prediction = predictor.predict(input_df)
        if prediction.iloc[0] == "sim":
            st.success(f"O resultado previsto para 'bolsista' é: {prediction.iloc[0]}")
            st.toast(f"#### O resultado previsto para 'bolsista' é: :green[{prediction.iloc[0]}]") 
        else:
            st.error(f"O resultado previsto para 'bolsista' é: {prediction.iloc[0]}")
            st.toast(f"#### O resultado previsto para 'bolsista' é: :red[{prediction.iloc[0]}]") 
        
        
        


def evaluate_model(predictor, test_data, X_test):
    y_test = test_data[predictor.label]
    y_pred = predictor.predict(X_test)
    perf = predictor.evaluate_predictions(
        y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
    )
    return y_test, y_pred, perf

def plot_confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    labels = ["não", "sim"]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predito", y="Verdadeiro", color="Contagem"),
    )
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Classe Predita",
        yaxis_title="Classe Verdadeira",
    )
    st.plotly_chart(fig)

def display_feature_importance(predictor, test_data):
    importance_df = predictor.feature_importance(test_data)
    importance_df = importance_df.reset_index().rename(columns={'index': 'Feature'})
    importance_df['Importance'] = importance_df['importance'].round(2)
    st.markdown("### Importância das Features")
    st.dataframe(importance_df[['Feature', 'Importance']])
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Importância das Features',
        color='Importance',
        color_continuous_scale='Blues',
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=400 + len(importance_df) * 20
    )
    st.plotly_chart(fig)

def main():
    data = load_data()
    data = preprocess_data(data)
    train_data, test_data, X_test = prepare_datasets(data)
    label = "bolsista"
    save_path = "data/agModels-predictBolsista"
    
    with st.sidebar:    
        with st.status("Treinando o Modelo ou Carregando Modelo Salvo - Bolsista"):
            with st.spinner('Processando...'):
                predictor = train_or_load_model(train_data, label, save_path)
            st.success('Modelo pronto!')
        
    st.sidebar.write("")

    predict_with_user_input(predictor)

