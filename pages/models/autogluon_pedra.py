# import pandas
import pandas as pd

# import train-test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

from autogluon.tabular import TabularDataset, TabularPredictor

import seaborn as sns

import streamlit as st

import matplotlib.pyplot as plt


# Lê arquivo de entrada de dados
data = pd.read_csv("../data/df_pooled_common.csv")

# converts 'ian' to category
data["ian"] = data["ian"].astype("category")

# converts 'sexo' to category
data["sexo"] = data["sexo"].astype("category")

# converts 'pedra' to category
data["pedra"] = data["pedra"].astype("category")

# converts 'ponto_virada' to category
data["ponto_virada"] = data["ponto_virada"].astype("category")

# converts 'fase' to category
data["fase"] = data["fase"].astype("category")

# converts 'na_fase' to boolean
data["na_fase"] = data["na_fase"].astype(bool)

# converts bolsista_encoded to boolean
data["bolsista_encoded"] = data["bolsista_encoded"].astype(bool)

# maps ano to t,t+1,t+2
data["ano"] = data["ano"].apply(
    lambda x: "t0" if x == 2020 else ("t1" if x == 2021 else "t2")
)
data["ano"] = data["ano"].astype("category")

data = data.drop(columns=["pedra_encoded", "nome", "corraca", "sexo_encoded"])

# Selecionar as variáveis preditoras e a variável alvo
# no lugar de eliminar o ano, será que daria para usar uma espécie de nota no ano t, nota no ano t+1, nota em t + 2?
X = data.drop(columns=["pedra", "inde"])

# features com importância
X = X[["ida", "ieg", "ipv", "ano", "iaa", "ian", "ipp", "ips", "na_fase"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, data["pedra"], test_size=0.25, random_state=41, shuffle=True
)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

label = "pedra"
#st.write("Summary of class variable: \n", train_data[label].describe())
print("Summary of class variable: \n", train_data[label].describe())

save_path = "agModels-predictPedra"  # specifies folder to store trained models


predictor = TabularPredictor(
    label=label, path=save_path, problem_type="multiclass"
).fit(train_data, presets="good_quality", num_gpus=1)

y_test = test_data[label]  # values to predict
test_data_nolab = X_test  # delete label column to prove we're not cheating


predictor = TabularPredictor.load(
    save_path
)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(
    y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
)

results = predictor.fit_summary(show_plot=False)

predictor.feature_importance(test_data)

features_importance = predictor.feature_importance(test_data)

important_features = features_importance[features_importance < 0.05].index.to_list()
print(important_features)

import seaborn as sns

cm = metrics.confusion_matrix(y_test, y_pred)

# Get the unique labels from y_test
labels = y_test.value_counts().index

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
