# Classificação de Estresse Acadêmico com KNN

##	Exploração dos Dados

A base possui **280 registros** e **9 colunas**.

*Estágio acadêmico*: Refere-se ao nível acadêmico do respondente (graduação, ensino médio, pós-graduação).

*Pressão dos colegas*: O quanto o aluno se sente pressionado pelos colegas em uma escala de 1 a 5.

*Pressão acadêmica da família*: O quanto o aluno se sente pressionado pela família em uma escala de 1 a 5.

*Ambiente de estudo*: Como o aluno classifica o ambiente onde estuda (categorias como *Peaceful*, *Noisy*, *Disrupted*).

*Estratégia de enfrentamento*: Como o aluno enfrenta o estresse (ex.: *Analisar a situação com inteligência*, *Colapso emocional*, *Apoio de amigos/família*).

*Maus hábitos*: Indica se o respondente possui hábitos nocivos, como fumar ou beber (Sim/Não/Prefiro não responder).

*Competição acadêmica*: Grau de competição acadêmica percebida (escala de 1 a 5).

*Índice de estresse*: Nível de estresse acadêmico (escala de 1 a 5).

---

##	Pré-processamento

#### Remoção de colunas irrelevantes
A coluna **Timestamp** foi removida por não agregar informações para a previsão.

#### Variável-alvo
A variável alvo é **Stress** (*índice de estresse acadêmico*, de 1 a 5).

#### Tratamento de valores ausentes
Foi identificado apenas um valor nulo em **StudyEnv** (ambiente de estudo), que foi preenchido com o valor mais frequente (*Peaceful*).

#### Codificação de variáveis categóricas
As colunas **AcademicStage**, **StudyEnv**, **Strategy** e **BadHabits** são categóricas e foram convertidas para valores numéricos utilizando **Label Encoding**.

#### Features e target

**features (X)**: estágio acadêmico, pressão dos colegas, pressão da família, ambiente de estudo, estratégia de enfrentamento, hábitos nocivos, nível de competição acadêmica.  
**target (y)**: índice de estresse acadêmico (1 a 5).

---

##	Divisão dos Dados
Os dados foram divididos de forma **estratificada** em **80% treino** e **20% teste**, garantindo que as proporções entre as classes de estresse fossem preservadas.

---

##	Treinamento do Modelo

=== "KNN"

    ```python exec="1" html="1"
    --8<-- "docs/KNN/student_knn.py"
    ```


=== "code"

    ```python exec="0"
    --8<-- "docs/KNN/student_knn.py"
    ```


##	Avaliação do Modelo


=== "Confusion Matrix"

    ```python exec="1" html="1"
    --8<-- "docs/KNN/confusionMatrix.py"
    ```


=== "code"

    ```python exec="0"
    --8<-- "docs/KNN/confusionMatrix.py"
    ```


##  Conclusão

O uso do **KNN** nesta base evidenciou pontos importantes para o aprendizado:

- O **desbalanceamento das classes** afeta a performance: níveis baixos de estresse (1 e 2) são pouco representados, o que dificulta acertos nessas classes.  
- O modelo se mostrou **sensível à escolha das variáveis de entrada** e ao valor de *k*. Testes com diferentes combinações podem alterar significativamente a acurácia.  
- A **normalização/standardização das variáveis numéricas** é fundamental para que todas as features tenham peso similar na distância euclidiana usada pelo KNN.  
- Técnicas como **oversampling (SMOTE)** ou até a **binarização do alvo** (baixo vs. alto estresse) podem ser boas alternativas dependendo do objetivo da análise.  
- A **interpretação do modelo** deve considerar o contexto: um erro em classificar estresse alto como baixo pode ter impacto maior do que o inverso, exigindo avaliação não apenas por acurácia, mas também por métricas como *recall* e *f1-score* em classes específicas.

Assim, o projeto reforça a importância do **pré-processamento cuidadoso**, da atenção ao **balanceamento de classes** e da análise crítica dos resultados para que o modelo seja útil de acordo com o problema real. 
