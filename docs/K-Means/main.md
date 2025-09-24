# Agrupamento de Estresse Acadêmico com K-Means

## Exploração dos Dados

A base `StressExp.csv` contém **280 registros** e **9 colunas**.  
As variáveis são relacionadas ao perfil acadêmico dos alunos, pressões percebidas, ambiente de estudo e hábitos.

- **AcademicStage**: Nível acadêmico do respondente.  
- **PeerPressure**: Pressão dos colegas (escala 1–5).  
- **HomePressure**: Pressão da família (escala 1–5).  
- **StudyEnv**: Ambiente de estudo (Peaceful, Noisy, Disrupted).  
- **Strategy**: Estratégias de enfrentamento.  
- **BadHabits**: Maus hábitos (Sim/Não).  
- **AcademicComp**: Competição acadêmica (escala 1–5).  
- **Stress**: Nível de estresse (1–5). Usado apenas como **validação externa**.  

```python exec="1"
--8<-- "docs/arvore-decisao/studentSample.py"
```

---

## Pré-processamento

- A coluna **Timestamp** foi removida por não agregar informação.  
- A variável **StudyEnv** apresentava 1 valor nulo, que foi preenchido com a moda (*Peaceful*).  
- Variáveis categóricas (**AcademicStage, StudyEnv, Strategy, BadHabits**) foram convertidas em valores numéricos via **Label Encoding**.  
- A variável **Stress** foi **removida das features** para o K-Means, sendo utilizada apenas na avaliação posterior.  
- As features numéricas foram **padronizadas** com `StandardScaler` para garantir que todas tenham a mesma escala.  


## Clustering K-Means

O modelo **K-Means** foi configurado com:  
- **k = 5 clusters** (escolhido por corresponder à escala de estresse 1–5).  
- Inicialização: `k-means++`.  
- Máximo de 100 iterações, `random_state=42`, `n_init=10`.  

### Resultados do Treinamento
- **Inércia (WCSS):** 1218.72  
- **Silhouette Score:** 0.19 → Indica que os clusters estão pouco separados e há sobreposição entre grupos.  

### Visualização (PCA 2D)

Os dados foram reduzidos para 2 dimensões com **PCA** apenas para visualização:  

=== "K-Means"

    ```python exec="1" html="1"
    --8<-- "docs/K-Means/student_Kmeans.py"
    ```


=== "code"

    ```python exec="0"
    --8<-- "docs/K-Means/student_Kmeans.py"
    ```


---

## Avaliação dos Clusters

A variável `Stress` foi usada para verificar como os clusters se alinham com os níveis reais de estresse.  

Tabela cruzada (**Stress x Cluster**):

| Stress | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|--------|-----------|-----------|-----------|-----------|-----------|
| **1**  | 13        | 6         | 9         | 7         | 21        |
| **2**  | 11        | 7         | 19        | 5         | 14        |
| **3**  | 3         | 26        | 8         | 15        | 4         |
| **4**  | 3         | 33        | 8         | 11        | 1         |
| **5**  | 9         | 30        | 3         | 13        | 1         |

Observa-se que os clusters não correspondem diretamente aos níveis de estresse, com forte sobreposição entre as classes.  

---

## Conclusão

- O **K-Means** não conseguiu separar claramente os grupos de alunos de acordo com o nível de estresse.  
- O **Silhouette Score baixo (0.19)** confirma que os clusters são pouco distintos.  
- Cada cluster mistura diferentes valores de estresse, mostrando que o algoritmo está agrupando mais por **perfil geral** (estágio acadêmico, hábitos, ambiente) do que pelo nível de estresse em si.  
- A análise é válida para identificar **padrões de perfis semelhantes**, mas não substitui modelos supervisionados como o KNN.  

### Recomendações
- Testar diferentes valores de **k** (método do cotovelo, silhouette).  
- **Normalizar e/ou reduzir dimensionalidade** antes do clustering (ex.: PCA completo).  
- Usar `Stress` apenas como **validação externa**, não como feature no clustering.  
- Considerar outras técnicas de clusterização (ex.: **DBSCAN, Agglomerative Clustering**) para comparação.  
