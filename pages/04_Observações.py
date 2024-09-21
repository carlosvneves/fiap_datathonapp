import streamlit as st

st.title("Observações 💡")

st.markdown("""
<div style= 'text-align: justify;'>

## Sobre a base de dados

#### Principais problemas:

- A série de dados disponibilizada é "curta" (apenas 3 anos) e "irregular" (ocorreram mudanças de metodologia entre os anos, ou seja, os dados foram coletados de forma diferente em cada ano);
- Foi necessário realizar vários ajustes para compatibilização das bases de dados, especialmente a transformação dos dados para o formato adequado para análise e construção dos modelos de _Machine Learning_;

#### Sugestões para melhoria:
- Coletar dados para mais anos, de modo que se possa realizar estudos mais amplos
e detalhados sobre o desempenho dos alunos na Passos Mágicos ao longo do tempo;
- Manter uma metodologia consistente para a coleta e tratamento dos dados, de modo que
seja necessário realizar menos transformações e ajustes na bases, os quais podem ser
mais uma fonte de erros e imprecisões;

## Sobre a análise exploratória de dados

- A análise de dados foi realizada com base na metodologia de análise longitudinal, onde
os dados foram coletados e analisados de forma contínua, com o objetivo de identificar
padrões e tendências nos dados ao longo do tempo;
- Como a série de dados disponibilizada é curta, não é possível ser assertivo quanto a
possíveis tendências;
- Destaca-se:
    - O INDE teve uma queda estatisticamente siginificativa em 2021, provavelmente como
    efeito da pandemia de COVID-19. Porém, em 2022 houve recuperação, com possível reversão de tendencia;
    - Não foi possível identificar diferença estatisticamente siginificativa na evolução do INDE em função
    do gênero dos alunos os anos de 2020 a 2023;
    - Houve aumento da proporção de alunos com pedras Ágata e Topázio entre 2020 e 2022;
    - Entre 2020 e 2022, houve redução da proporção de alunos com bolsas de estudos em 2020, mas com umaumento em 2021;
    - Na análise  transversal, há o detalhamento de alguns fatores para cada ano, como por exemplo, 
    a quantidade de bolsas de estudos, o gênero dos alunos, entre outros.

## Sobre os modelos de _Machine-Learning_

- Os modelos de _Machine Learning_ foram treinados e testados em dados de 2020, 2021 e 2022, 
os quais foram pré-processados para que possam ser utilizados como dados de treinamento e de teste.
- Para a **Classificação Geral** foi desenvolvido um modelo de clusterização hieráquica com Random Forest, 
de modo que é possível perceber a estrutura subjacente aos dados e identificar os padrões de agrupamento 
das diversas variáveis. Então se pode entender como os fatores que influenciam na classificação geral do aluno na Passos Mágicos.
- Para a **Pedra-Conceito** foi desenvolvido um **modelo de classificação do tipo _WeightedEnsemble_L3_**. 
Este modelo apresentou resultados satisfatórios aoclassificar os alunos de acordo com sua Pedra-Conceito. 
- Um **modelo de classificação do tipo _WeightedEnsemble_L1_** também foi desenvolvido para permitir melhor compreensão e 
previsão da **Concessão de Bolsa de Estudos**. 

## Conclusões

- A análise de dados foi realizada com base na metodologia de análise exploratória de dados, onde os dados foram coletados e 
analisados de forma contínua, com o objetivo de identificar padrões e tendências nos dados ao longo do tempo;
- Foram também construídas interfaces gráficas para facilitar a visualização dos dados e a compreensão dos resultados;
- Espera-se que o projeto seja útil para a Associação Passos Mágicos e que possa ser utilizado como base para futuras análises.

### Ferramentas e tecnologias utilizadas
- O projeto foi desenvolvido em [Python 3.11.9](python.org);
- As principais bibliotecas utilizadas foram:
    - [_Streamlit_](streamlit.io): biblioteca para criação da interface gráfica;
    - [_Pandas_](pandas.pydata.org): biblioteca para manipulação de dados;
    - [_Plotly Express_](plotly.com/python/): biblioteca para criação de gráficos;
    - [_YData Profiling_](docs.profilling.ydata.ai): biblioteca para visualização dos dados;
    - [_Autogluon_](auto.gluon.ai): biblioteca para treinamento e validação de modelos de _Machine Learning_;
    - [_scikit-learn_](scikit-learn.org): biblioteca para treinamento e validação de modelos de _Machine Learning_;
    - [_scipy_](scipy.org): biblioteca para cálculo de estatísticas;
    - [_statsmodels_](statsmodels.org): biblioteca para cálculo de estatísticas;
    - [_seaborn_](seaborn.pydata.org): biblioteca para visualização de gráficos;
    - [_nltk_](nltk.org): biblioteca para tokenização de textos e análise de sentimentos;
</div>""",
    unsafe_allow_html=True,
)
