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
    # Select important features
    X = X[["ida", "ieg", "ipv", "ano", "iaa", "ian", "ipp", "ips", "na_fase"]]
    y = data["pedra"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=41, shuffle=True
    )
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    return train_data, test_data, X_test

def train_or_load_model(train_data, label, save_path):
    # Check if the model exists at save_path and is not empty
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

def evaluate_model(predictor, test_data, X_test):
    y_test = test_data[predictor.label]
    test_data_nolab = X_test.copy()
    y_pred = predictor.predict(test_data_nolab)
    perf = predictor.evaluate_predictions(
        y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
    )
    return y_test, y_pred, perf

def plot_confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
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
        st.dataframe(importance_df[['Feature', 'Importance']], hide_index=True)

def main():
    data = load_data()
    data = preprocess_data(data)
    train_data, test_data, X_test = prepare_datasets(data)
    label = "pedra"
    save_path = "data/agModels-predictPedra"
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
                    #### O modelo é capaz de identificar a Pedra-Conceito de forma muito precisa. 
                    Isto se reflete na **Matriz de Confusão**, a qual mostra que o modelo identifica 
                    a Pedra-Conceito corretamente com **{perf['accuracy']*100:.2f}%** de acurácia.            
                    
                    Entre as variáveis escolhidas para o modelo, as três mais importantes são:
                    - Indicador de Desempenho Acadêmico (IDA);
                    - Indicador de Engajamento (IEG); e
                    - Indicador do Ponto de Virada (IPV).                
                    """)    
        
    st.divider()

    display_feature_importance(predictor, test_data)
    st.divider()
        
    plot_confusion_matrix(y_test, y_pred)
        
    st.divider()        
    st.markdown("#### Resumo do Modelo")
    with st.expander("Ver base de dados usada para treinar e testar o modelo"): 
        st.dataframe(data, hide_index=True)
    with st.expander("Ver detalhes dos modelos testados usando a biblioteca `Autogluon`"): 
        results = predictor.fit_summary(show_plot=False)
        st.text(results)


