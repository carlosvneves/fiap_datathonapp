import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn import metrics
import plotly.express as px
import os

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
        xaxis_title="Rótulo Previsto",
        yaxis_title="Rótulo Verdadeiro",
    )
    st.plotly_chart(fig)

def display_feature_importance(predictor, test_data):
    importance_df = predictor.feature_importance(test_data)
    importance_df = importance_df.reset_index().rename(columns={'index': 'Feature'})
    importance_df['Importance'] = importance_df['importance'].round(2)
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
    with st.expander("Mostrar tabela"):
        st.dataframe(importance_df[['Feature', 'Importance']])


def main():
    data = load_data()
    data = preprocess_data(data)
    train_data, test_data, X_test = prepare_datasets(data)
    label = "bolsista"
    save_path = "data/agModels-predictBolsista"
    st.markdown("#### Treinando o Modelo ou Carregando Modelo Salvo")
    with st.spinner('Processando...'):
        predictor = train_or_load_model(train_data, label, save_path)
    st.success('Modelo pronto!')
    y_test, y_pred, perf = evaluate_model(predictor, test_data, X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Acurácia", f"{perf['accuracy']*100:.2f}%")
        st.metric("Acurácia Balanceada", f"{perf['balanced_accuracy']*100:.2f}%")
    with col2:
        st.markdown(f""" 
                    #### O modelo é capaz de identificar corretamente se o estudante, está em condições de receber ou não a bolsa de estudos. 
                    
                    Isto se reflete na **Matriz de Confusão**, a qual mostra que o modelo identifica 
                    corretamente quem recebe a bolsa com uma acurácia de **{perf['accuracy']*100:.2f}%**.            
                    
                    Entre as variáveis escolhidas para o modelo, as três mais importantes são:
                    - Fase;
                    - Anos na Passos Mágicos; e
                    - Indicador de Desempenho Acadêmico (IDA).              
                                    
                    """)
    
    st.divider()
    #st.markdown("### Importância das Features")
    display_feature_importance(predictor, test_data)
    st.divider()
    #st.markdown("### Matriz de Confusão")
    plot_confusion_matrix(y_test, y_pred)
    
    st.divider()            
        
    st.markdown("### Resumo do Modelo")
    with st.expander("Ver base de dados usada para treinar e testar o modelo"): 
        st.dataframe(data, hide_index=True)
    with st.expander("Ver detalhes dos modelos testados usando a biblioteca `Autogluon`"): 
        results = predictor.fit_summary(show_plot=False)
        st.text(results)   