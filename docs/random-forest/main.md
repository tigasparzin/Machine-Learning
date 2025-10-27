# Análise – Índice de Estresse Acadêmico (Random Forest)

##	Exploração dos Dados

A base possui **280 registros** e **9 colunas** (`Timestamp`, `AcademicStage`, `PeerPressure`, `HomePressure`, `StudyEnv`, `Strategy`, `BadHabits`, `AcademicComp`, `Stress`).

**Distribuições e estatísticas principais:**

- **Estágio acadêmico** (`AcademicStage`): undergraduate **63.57%**, high school **22.14%**, post-graduate **14.29%**.
- **Pressão dos colegas** (`PeerPressure`): média **3.01** (escala 1–5).
- **Pressão acadêmica da família** (`HomePressure`): média **3.09** (1–5).
- **Ambiente de estudo** (`StudyEnv`): Peaceful **47.14%**, Noisy **26.43%**, disrupted **26.07%**, e **0.36%** ausente (1 valor nulo).
- **Estratégia de enfrentamento** (`Strategy`): Analyze the situation… **54.29%**, Emotional breakdown **26.79%**, Social support **18.93%**.
- **Maus hábitos** (`BadHabits`): No **65.71%**, Yes **18.93%**, prefer not to say **15.36%**.
- **Competição acadêmica** (`AcademicComp`): média **3.27** (1–5).
- **Índice de estresse (target)** (`Stress`): **balanceado** — 20.0% (1), 20.0% (2), 20.0% (3), 20.0% (4), 20.0% (5); média **3.0**.

```python exec="1"
--8<-- "docs/arvore-decisao/studentSample.py"
```

---

##	Pré-processamento

#### Remoção de colunas irrelevantes
A coluna `Timestamp` foi removida por não agregar informação para a previsão.

#### Variável-alvo
A variável alvo definida foi `Stress` (índice de estresse acadêmico: 1 a 5).

#### Tratamento de missing value
A base apresenta **1** valor nulo em `StudyEnv`. Preencher com **Peaceful** (moda) **calculada somente nos dados de treino** .

#### Codificação de variáveis categóricas
As colunas `AcademicStage`, `StudyEnv`, `Strategy` e `BadHabits` são categóricas.
O pipeline atual usa **Label Encoding** — válido para modelos de árvore.

#### Features e target
- **features (X):** `AcademicStage`, `PeerPressure`, `HomePressure`, `StudyEnv`, `Strategy`, `BadHabits`, `AcademicComp`
- **target (y):** `Stress` (1 a 5)

##	Divisão dos Dados
80% dos registros foram separados para treino e 20% para teste.  

## Binarização
Aproveitei o modelo KNN feito anteriormente e mantive a binarização do target neste modelo.
(niveis de estresse menores ou iguais a 3 são classificados como "baixo" e maiores que 3 como "alto"), porém após rodar o modelo, acabou resultando num overfitting (97% de acurácia), assim, a binarização foi revertida.

##	Treinamento do Modelo

=== "random forest code"

    ```python exec="0"
    --8<-- "docs/random-forest/randomForest.py"
    ```

##	Avaliação do Modelo

No modelo **Random Forest**, a **acurácia** foi de **~0.66** .  


=== "avaliacao do modelo"

    ```python exec="1" html="1"
    --8<-- "docs/random-forest/avaliacao.py"
    ```
=== "code"

    ```python exec="0"
    --8<-- "docs/random-forest/avaliacao.py"
    ```


## Conclusão




