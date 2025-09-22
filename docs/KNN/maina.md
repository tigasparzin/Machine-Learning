##	Exploração dos Dados
Para esse projeto, utilizarei a base ja tratada usada no exemplo da arvore de decisao

A base possui 280 registros e 9 colunas.

*Estágio acadêmico*: Refere-se ao nível acadêmico do respondente. A distribuição é composta por 63,6% estudantes de graduação, 22,1% do ensino médio e 14,3% de pós-graduação

*Pressão dos colegas*: Mede o quanto o aluno se sente pressionado pelos colegas em uma escala de 1 a 5. A média das respostas foi de 3,01.

*Pressão acadêmica da família*: Mede o quanto o aluno se sente pressionado pela família em uma escala de 1 a 5. A média das respostas foi de 3,09.

*Ambiente de estudo*: Classificação do ambiente onde o aluno estuda. As respostas foram: Peaceful (47,3%), Noisy (26,5%) e Disrupted (26,2%).

*Estratégia de enfrentamento*: Representa como o aluno lida com o estresse. As respostas foram: 54,3% analisam a situação com inteligência, 26,8% recorrem a colapso emocional e 18,9% buscam apoio de amigos/família.

*Maus habitos*: Indica se o respondente possui hábitos como fumar ou beber. 65,7% responderam "Não", 18,9% "Sim" e 15,4% preferiram não responder.

*Competição acadêmica*: Mede a competição acadêmica percebida na vida do estudante em uma escala de 1 a 5. A média foi de 3,27.

*Índice de estresse*: Classificação do nível de estresse acadêmico em uma escala de 1 a 5. A média foi de 3,00.

---

##	Pré-processamento

#### Remoção de colunas irrelevantes
A coluna Timestamp foi removida por não agregar informações para a previsão.

#### tratamento de missing values
Assim como na decision tree, o valor faltante foi definido como "Peaceful", valor que mais aparece

#### Variável-alvo
A variável alvo definida foi "Stress".

#### Codificação de variáveis categóricas
As colunas "AcademicStage", "StudyEnv", "Strategy" e "BadHabits", são categóricas.

Foram convertidas em valores numéricos utilizando Label Encoding, para que a árvore de decisão consiga interpretá-las.

#### Redução de dimensoes(PCA)
Para facilitar a visualização dos dados e da fronteira de decisão do modelo, apliquei a técnica de análise de componentes principais, reduzindo as variáveis originais para duas dimensões. permitindo representar graficamente como o KNN separa as classes no plano.

#### Features e target
feature (X): estágio acadêmico, pressão dos colegas, pressão da família, ambiente de estudo, estratégia de enfrentamento, hábitos nocivos, nível de competição acadêmica.

target (y): índice de estresse acadêmico (1 a 5).


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


##  Conclusão