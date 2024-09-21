import streamlit as st
st.title("Apresentação")
st.markdown("""
## Qual problema se pretende solucionar por meio deste _WebApp/Dashboard_?
O _Datathon_ proposto aos alunos da turma de Pós-Tech em Data Analytics da FIAP (3DTAT) trata da construção de um _Dashboard_ para visualização e compreensão dos dados da PEDE - Pesquisa Extensiva do Desenvolvimento Educacional dos alunos da **Associação Passos Mágicos** -, e o desenvolvimento de Modelos Preditivos que possam auxiliar no planejamento das ações da instituição. 
""")
st.markdown("""
## O que é a Associação Passos Mágicos?
Segundo a página oficial da própria ONG [Passos Mágicos](https://passosmagicos.org.br/):
> A Associação Passos Mágicos tem uma trajetória de 30 anos de atuação, trabalhando na transformação da vida de crianças e jovens de baixa renda os levando a melhores oportunidades de vida.
A transformação, idealizada por Michelle Flues e Dimetri Ivanoff, começou em 1992, atuando dentro de orfanatos, no município de Embu-Guaçu.
Em 2016, depois de anos de atuação, decidem ampliar o programa para que mais jovens tivessem acesso a essa fórmula mágica para transformação que inclui: educação de qualidade, auxílio psicológico/psicopedagógico, ampliação de sua visão de mundo e protagonismo. Passaram então a atuar como um projeto social e educacional, criando assim a Associação Passos Mágicos.
Na Associação são atendidas crianças e jovens oriundos de escolas públicas do município de Embu-Guaçu, mas no contexto de todas as ações empreendidas pela associação é desenvolvido um intenso programa de bolsas de estudo em instituições privadas de ensino. Esse programa de bolsas abrange tanto o ensino fundamental como o ensino médio, e também atende as necessidade de alguns alunos no seu ingresso no ensino superior. Todos os bolsistas do ensino fundamental e médio mantém sua rotina de estudos no Programa de Aceleração do Conhecimento, e os jovens bolsistas do ensino superior se mantém engajados nos processos de avaliação da associação, bem como em suas ações de voluntariado. 
""")
st.markdown("""
## O que é a PEDE da Associação Passos Mágicos?

A PEDE Passos Mágicos, ou Pesquisa Extensiva do Desenvolvimento Educacional, é uma avaliação educacional realizada pela Associação Passos Mágicos anualmente, é uma iniciativa voltada para monitorar e avaliar o progresso educacional dos beneficiados pelos seus programas. A PEDE fornece dados e informações sobre o desempenho dos alunos, não apenas em termos acadêmicos e psicossociais.
**Principais Objetivos:**
* **Avaliar o Impacto da Associação:**  A PEDE busca determinar a influência das atividades da Passos Mágicos na vida educacional de seus alunos, utilizando uma medida de "IMPACTO" para mensurar as mudanças no desempenho ao longo do tempo.
* **Identificar Áreas de Aperfeiçoamento:**  A análise dos dados permite identificar os pontos fortes e fracos do programa educacional, oferecendo insights para aprimorar as estratégias de ensino e aprendizagem.
* **Promover a Equidade:** A pesquisa investiga a progressão dos alunos, considerando seu histórico como bolsistas ou estudantes de escola pública, e se realizaram todas as provas, contribuindo para a construção de um ambiente educacional mais justo e igualitário.

 """)

st.markdown("""

## Qual é a metodologia usada pela PEDE?
A metodologia da PEDE envolve uma combinação de diferentes indicadores que visam permitir uma visão holística sobreo desempenho e bem-estar dos alunos. Os diversos indicadores compõem um índice denominado INDE (Índice de Desenvolvimento Educacional), o qual é uma medida síntese do processo avaliativo da PEDE. Para tanto, foram selecionados os critérios de mérito das ações da Associação, representadas pelas dimensões de avaliação (Acadêmica, Psicossocial e Psicopedagógica). Em seguida, foram definidos padrões de desempenho, os indicadores, por meio dos quais essas dimensões são observadas e medidas.
As dimensões Acadêmica, Psicossocial e Psicopedagógica são medidas por meio dos seguintes indicadores:

---
**1. Dimensão Acadêmica**:
- **Indicador de Adequação de Nível (IAN)**: verifica se o aluno está na fase de ensino correta de acordo com sua idade. Ele mede a defasagem entre a idade do aluno e o ano escolar em que ele se encontra, sendo um indicador importante para medir o progresso ao longo dos anos.
- **Indicador de Desempenho Acadêmico (IDA)**: mede o desempenho dos alunos em disciplinas centrais, como Matemática, Português e Inglês. Essas notas acadêmicas são padronizadas e usadas para refletir o sucesso escolar em termos de aprendizagem formal. O IDA contribui diretamente para o INDE, representando o progresso cognitivo e acadêmico do aluno, abrangendo desde a Fase 0 (alfabetização), até a Fase 7 (3º Ano do Ensino Médio).
- **Indicador de Engajamento (IEG)**: mede o quanto o aluno participa ativamente das atividades escolares e extracurriculares, incluindo sua interação com colegas e professores. O nível de engajamento é crucial, pois alunos mais engajados tendem a ter melhores resultados a longo prazo. Esse indicador ajuda a entender o quanto o ambiente escolar e as atividades extracurriculares estão impactando positivamente a educação.

---
**2. Dimensão Psicossocial**: 
- **Indicador de Autoavaliação (IAA)**: reflete a percepção do aluno sobre seu próprio desempenho e habilidades. Essa autoavaliação é fundamental para entender a autoconfiança do estudante em relação à aprendizagem. Um alto IAA pode indicar que o aluno acredita estar bem preparado, enquanto um IAA baixo pode sugerir necessidade de suporte adicional.
4. **Indicador Psicossocial (IPS)**: avalia a saúde emocional e social do aluno, medindo aspectos como resiliência, capacidade de enfrentar desafios e interações sociais. Esse indicador é importante porque, muitas vezes, fatores emocionais têm um impacto significativo no desempenho acadêmico e na permanência escolar.

---
**3. Dimensão Psicopedagógica**:
- **Indicador Psicopedagógico (IPP)**: foca no suporte pedagógico oferecido aos alunos, avaliando a eficácia das intervenções educativas. Alunos que recebem apoio pedagógico adequado tendem a apresentar melhor desempenho acadêmico e desenvolvimento geral.
- **Indicador do Ponto de Virada (IPV)**: representa momentos chave no desenvolvimento acadêmico ou social do aluno, como quando um aluno supera uma dificuldade significativa ou atinge um marco importante em sua educação. Esse indicador pode mostrar mudanças críticas que afetam o progresso do aluno.

Os indicadores acima ainda podem ser divididos em **Indicadores de Avaliação** (IAN, IDA, IEG e IAA) e **Indicadores de Conselho** (IPS, IPP e IPV). 

#### Pedra, outro conceito importante 

O INDE (Índice de Desenvolvimento Educacional) é uma medida agregada que resume o desempenho geral dos alunos, combinando os diferentes indicadores individuais (IAN, IDA, IEG, IAA, IPS, IPP e IPV). O INDE visa fornecer uma visão ampla da evolução dos estudantes ao longo do tempo, oferecendo uma base sólida para intervenções pedagógicas e sociais.

Dentro do sistema da PEDE, os alunos também são categorizados em níveis chamados **"Pedra"**, que são:
- Topázio,
- Ametista,
- Ágata,
- Quartzo.

Esses níveis funcionam como classificações de desempenho geral, sendo **"Topázio"** o mais elevado, e **"Quartzo"**, o nível que indica necessidade de maior suporte por parte da associação. As mudanças no nível da Pedra ao longo do tempo refletem o progresso ou a necessidade de intervenções pedagógicas mais intensivas.
A Pedra é calculada a partir da padronização do INDE individual dos alunos - análise relativa -, com a definição de intervalos ("Pedras") para melhor orientar as intervenções e análises por grupo de alunos.

""")
