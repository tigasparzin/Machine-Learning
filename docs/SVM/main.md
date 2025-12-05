# Análise – Índice de Estresse Acadêmico (SVM com kernels)

## Exploração dos Dados

A base **StressExp** possui **280 registros** e **9 colunas** (`Timestamp`, `AcademicStage`, `PeerPressure`, `HomePressure`, `StudyEnv`, `Strategy`, `BadHabits`, `AcademicComp`, `Stress`).

Para o experimento com SVM, o código utiliza apenas duas variáveis numéricas como features:

- **PeerPressure** (pressão dos colegas – escala 1–5)  
- **HomePressure** (pressão acadêmica da família – escala 1–5)

A variável original **Stress** é um índice de estresse de 1 a 5, aproximadamente balanceado nos 5 níveis, com média próxima de 3, como identificado anteriormente.

Esse recorte reduz o problema a um plano 2D, permitindo a visualização das fronteiras de decisão, mas ignorando variáveis relevantes como `AcademicComp`, `StudyEnv`, `BadHabits`, etc.

---

## Pré-processamento

### Colunas utilizadas
O modelo SVM usa apenas:

- `PeerPressure`
- `HomePressure`

### Variável-alvo (binarização)
O target é reduzido de cinco níveis para duas classes:

- `Stress >= 4` → **alto estresse**  
- `Stress <= 3` → **baixo estresse**  

Na implementação:

```
0 = alto estresse  
1 = baixo estresse
```

### Missing values
Nenhum dos dois atributos utilizados apresenta valores nulos, portanto nenhuma imputação foi necessária.

### Normalização
Não há normalização no código original, o que pode prejudicar kernels como RBF, sigmoid e poly.

---

## Divisão dos Dados

O código NÃO separa treino e teste:  
o modelo é treinado e visualizado na própria base completa.

Isso impede a avaliação real de desempenho, tornando o experimento **puramente visual/exploratório**.

---

## Treinamento do Modelo

O script treina quatro SVMs com `C = 1`, variando o kernel:

- **Linear**
- **Sigmoid**
- **Poly**
- **RBF**

=== "avaliacao do modelo"

    ```python exec="1" html="1"
    --8<-- "docs/SVM/SVM.py"
    ```
=== "code"

    ```python exec="0"
    --8<-- "docs/SVM/SVM.py"
    ```


Após treinar, o código plota as regiões de decisão em 2D, permitindo comparação visual.


## Avaliação do Modelo

Como não há métricas, apenas observações qualitativas podem ser feitas:

- Usar duas features limita severamente o poder do modelo.  
- Kernels não lineares provavelmente mostram melhor separação visual.  



## Conclusão

- O SVM binário é mais coerente com a estrutura do algoritmo.  
- O experimento atual é útil para **visualização didática** das fronteiras de decisão.  
- Para uma análise robusta, seria melhor utilizar normalização e mais atributos.

