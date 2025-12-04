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

Após treinar, o código plota as regiões de decisão em 2D, permitindo comparação visual.

### Comportamento esperado dos kernels

- **Linear** → fronteira simples; pode underfittar se as classes não forem lineares.  
- **Poly** → mais flexível; pode overfittar dependendo do grau.  
- **RBF** → geralmente o mais robusto; cria regiões suaves ao redor das classes.  
- **Sigmoid** → instável sem normalização; costuma performar abaixo do RBF.

---

## Avaliação do Modelo

Como não há métricas, apenas observações qualitativas podem ser feitas:

- Usar duas features limita severamente o poder do modelo.  
- A binarização ajuda o SVM, pois a tarefa vira binária (caso ideal do algoritmo).  
- Kernels não lineares provavelmente mostram melhor separação visual.  
- Sem normalização e sem teste, não é possível afirmar desempenho real.

Para um modelo comparável ao Random Forest, seria necessário incluir:

- `train_test_split`
- `StandardScaler`
- métricas (accuracy, F1, matriz de confusão)
- todas as variáveis relevantes

---

## Conclusão

- O SVM binário é mais coerente com a estrutura do algoritmo.  
- O experimento atual é útil para **visualização didática** das fronteiras de decisão.  
- Não deve ser interpretado como modelo final ou validado.  
- Para uma análise robusta, recomenda-se incluir métricas, normalização e mais atributos.

