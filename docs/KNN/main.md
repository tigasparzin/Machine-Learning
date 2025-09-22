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
A coluna **Timestamp** é removida por não agregar informações para a previsão.

#### Variável-alvo
A variável alvo é **Stress** (*índice de estresse acadêmico*, de 1 a 5).

#### Tratamento de valores ausentes
Se existir valor nulo em **StudyEnv** (ambiente de estudo), preencher com o **moda** (geralmente “Peaceful”).

#### Codificação de variáveis categóricas
As colunas **AcademicStage**, **StudyEnv**, **Strategy** e **BadHabits** são categóricas e são convertidas para valores numéricos via **Label Encoding**.

#### Features e target

**features (X)**: estágio acadêmico, pressão dos colegas, pressão da família, ambiente de estudo, estratégia de enfrentamento, hábitos nocivos, nível de competição acadêmica.  
**target (y)**: índice de estresse acadêmico (1 a 5).

---

##	Divisão dos Dados
Separação **estratificada** em **80% treino** e **20% teste**, preservando a proporção das classes do alvo.

---

##	Treinamento do Modelo

=== "KNN"

    ```python exec="1" html="1"
    --8<-- "docs/KNN/student_KNN.py"
    ```


=== "code"

    ```python exec="0"
    --8<-- "docs/KNN/student_KNN.py"
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

O **KNN** é simples e eficiente quando há uma boa representação de vizinhança no espaço de atributos. Neste caso, o **desbalanceamento das classes** do *Stress* influencia as métricas e pode reduzir a acurácia. Ajustes como **balanceamento**, **normalização**, **escolha adequada de _k_** e, quando fizer sentido, **binarização do alvo**, tendem a melhorar os resultados. A interpretação dos achados deve considerar o objetivo do projeto e o custo de erros por classe (por exemplo, confundir estresse alto com baixo).

