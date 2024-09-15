import streamlit as st

st.title("Análise Exploratória")

st.markdown("""

## Sobre a base de dados

Foi fornecido aos alunos da Pós-Tech um conjunto de dados anonimizados sobre a PEDE dos anos de 2020, 2021 e 2022, sobre mais de 1.300 alunos pesquisados na PEDE. Esses dados por aluno são tais como:
- Idade, sexo, cor/raça;
- INDE e os indicadores que o compõe;
- Anos na Passos Mágicos;
- Fase e Turma;
- Se o aluno é ou não bolsista;
- Classificações do aluno no ranking geral, por turma e por fase;

Os dados passaram por um processo de limpeza e tratamento, uma vez que, de acordo com o ano da pesquisa, foram coletados e preenchidos de modo diferente, refletindo certas mudanças metodológicas na pesquisa ao longo dos anos. Por exemplo, há diversas colunas que existem em um determinado ano, porém não existem em outro. Em casos como esse, quando possível, optou-se por realizar a imputação com base em outras colunas relacionadas.

Em alguns casos outros tipos de tratamento foram aplicados aos dados, como no caso de colunas contendo dados textuais (com avaliação sobre os alunos), os quais tiveram que ser submetidos a um tratamento por meio de processamento de linguagem natural, de modo que pudessem se tornar informativos o suficiente. Em outros casos foi aplicada a estratégia de _encoding_ para utilização do dado dentro de um modelo.

Quando necessário para a compreensão de determinado gráfico, o tratamento aplicado ao dado e o seu significado serão devidamente explicitados. 

""")


st.markdown("""
---
## Visão Geral



""")
