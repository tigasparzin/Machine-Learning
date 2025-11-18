# Análise de PageRank na rede de confiança soc-Epinions1

## 1. Objetivo

O objetivo deste trabalho é implementar do zero o algoritmo PageRank em um grafo dirigido real, comparar os resultados com a implementação pronta do NetworkX e analisar os nós mais importantes da rede em função de diferentes valores do fator de amortecimento \(d\).  

O dataset escolhido foi **soc-Epinions1**, que representa uma rede de confiança entre usuários do site de reviews Epinions. Em termos conceituais:

- **Nó** = usuário da plataforma;  
- **Aresta dirigida A → B** = o usuário A **confia** no usuário B.

A aplicação do PageRank, nesse contexto, permite identificar usuários que são não apenas muito confiados diretamente, mas também confiados por outros usuários igualmente confiáveis – ou seja, potenciais **influenciadores** na comunidade.

---

## 2. Dataset e modelagem do grafo

O arquivo de arestas `soc-Epinions1.txt` foi carregado diretamente como um grafo dirigido:

- Biblioteca utilizada: `networkx` (`nx.DiGraph()`);  
- Comentários iniciados por `#` foram ignorados;  
- IDs de nós foram lidos como inteiros.

Após o carregamento:

- Número de nós: **75.879**  
- Número de arestas dirigidas: **508.837**

Modelagem adotada:

- Uma aresta A → B significa que **A confia em B**;  
- O grafo é mantido como dirigido (não foi simetrizado).

Essa modelagem é coerente com a interpretação original da rede: a confiança é assimétrica (eu posso confiar em você sem que você necessariamente confie em mim).

---

## 3. Implementação do PageRank do zero

### 3.1. Estrutura de dados

Para evitar uma matriz de transição \(N \times N\) (inviável para ~76k nós), foi usada uma representação baseada em listas de adjacência:

- `nodes`: lista de nós (IDs inteiros)  
- `node_to_idx`: mapeia ID do nó → índice [0..N-1]  
- `out_neighbors[i]`: lista de índices dos nós alcançados por arestas que saem do nó de índice `i`;  
- `out_degree[i]`: grau de saída de cada nó.

Esses vetores permitem que o algoritmo rode em \(O(N + E)\) por iteração, sem explodir memória.

### 3.2. Fórmula iterativa

A implementação segue a fórmula clássica:

\[
PR(p_i) = \frac{1-d}{N} 
+ d \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)}
\]

Onde:

- \(d\) é o fator de amortecimento (teletransporte);  
- \(N\) é o número de nós;  
- \(M(p_i)\) é o conjunto de nós que apontam para \(p_i\);  
- \(L(p_j)\) é o número de ligações de saída de \(p_j\).

**Tratamento de nós “dangling” (sem saídas)**  
Nós com grau de saída zero acumulam PageRank que não é redistribuído via arestas. Para corrigir isso, foi usada a estratégia padrão:

- Soma-se a “massa” de PageRank de todos os nós sem saída (`dangling_sum`) e redistribui-se igualmente entre todos os nós, multiplicada por \(d\).

### 3.3. Critério de convergência

- Inicialização: \(PR_i^{(0)} = 1/N\) para todos os nós;  
- A cada iteração, é calculado um novo vetor `pr_new`;  
- Métrica de convergência:  
  \[
  \max_i |PR_i^{(t+1)} - PR_i^{(t)}|
  \]
- Se essa diferença máxima ficar abaixo de `tol = 1e-6`, o algoritmo para;  
- Limite de iterações: `max_iter = 100`.

---

## 4. Validação contra `networkx.pagerank`

Para **\(d = 0{,}85\)** (valor padrão clássico), os resultados da implementação própria foram comparados com `nx.pagerank(G, alpha=0.85)`:

- Diferença L1 total (soma dos módulos das diferenças):  
  \[
  \approx 8{,}88 \times 10^{-4}
  \]
- Diferença máxima entre qualquer nó:  
  \[
  \approx 1{,}93 \times 10^{-5}
  \]

O algoritmo próprio convergiu em **31 iterações** para \(d = 0{,}85\).

Esses valores de diferença são muito pequenos e estão dentro do esperado para métodos iterativos com tolerâncias ligeiramente diferentes, o que indica que a implementação está **correta e consistente** com a referência do NetworkX.

---

## 5. Resultados: nós mais importantes

Foram avaliados três valores de fator de amortecimento:

- \(d = 0{,}50\)  
- \(d = 0{,}85\)  
- \(d = 0{,}99\)

### 5.1. Convergência

- **\(d = 0{,}50\)**  
  - Convergência em **7 iterações**.  
- **\(d = 0{,}85\)**  
  - Convergência em **31 iterações**.  
- **\(d = 0{,}99\)**  
  - Não convergiu dentro de 100 iterações para `tol = 1e-6` (ou seja, precisaria de mais iterações ou tolerância mais frouxa).

Isso já mostra um comportamento importante: **quanto maior o \(d\)** (menor peso do teletransporte), **mais lenta a convergência**.

### 5.2. Top 10 nós para \(d = 0{,}85\)

A tabela abaixo resume os 10 nós com maior PageRank para \(d = 0{,}85\), com respectivos graus de entrada/saída:

| Posição | Nó   | PageRank        | Grau de entrada | Grau de saída |
|--------|------|-----------------|-----------------|---------------|
| 1      | 18   | 4.535e-03       | 3035            | 44            |
| 2      | 737  | 3.151e-03       | 1317            | 372           |
| 3      | 118  | 2.122e-03       | 1004            | 123           |
| 4      | 1719 | 2.078e-03       | 1140            | 46            |
| 5      | 136  | 1.987e-03       | 1180            | 111           |
| 6      | 790  | 1.969e-03       | 1284            | 102           |
| 7      | 143  | 1.957e-03       | 1521            | 171           |
| 8      | 40   | 1.825e-03       | 817             | 238           |
| 9      | 1619 | 1.536e-03       | 784             | 123           |
| 10     | 725  | 1.496e-03       | 694             | 274           |

Observações:

- Todos esses nós apresentam **graus de entrada muito altos** (centenas a milhares de usuários que confiam neles);  
- Alguns também têm graus de saída elevados (como os nós 737, 40 e 725), o que sugere que eles também confiam em muitos outros, atuando como “hubs” de confiança.

### 5.3. Top 10 nós para outros valores de \(d\)

**Para \(d = 0{,}50\)**, o top 10 é muito parecido com o de \(d = 0{,}85\):

- Nó 18 continua em 1º lugar;  
- Nós como 737, 790, 1719, 143, 136 e 118 aparecem nos primeiros lugares;  
- As diferenças são pequenas na ordenação e na concentração dos scores.

**Para \(d = 0{,}99\)**:

- O nó 18 continua líder (PR ainda maior);  
- 737 permanece no topo;  
- Surgem nós como **22933** e **8648** com PageRank bem alto apesar de terem graus de entrada modestos (23 e 102) e **grau de saída 1**.  
  - Isso sugere que eles podem estar recebendo links de nós muito influentes ou fazendo parte de “caminhos” de confiança que prendem a massa de PageRank quando o teletransporte é raro.

---

## 6. Interpretação no contexto da rede Epinions

Mesmo sem metadados (nomes reais dos usuários, temas de reviews, etc.), é possível interpretar qualitativamente:

1. **Nós com altíssimo grau de entrada e PageRank alto**  
   - Ex.: nós 18, 143, 790, 118, 136, 1719;  
   - São usuários que recebem confiança de muitos outros, incluindo possivelmente outros usuários influentes;  
   - Em um site de reviews, podem ser:
     - “Top reviewers”;  
     - Usuários antigos com histórico confiável;  
     - Pessoas centrais em comunidades temáticas específicas.

2. **Relação entre grau e PageRank**  
   - PageRank não é apenas “contar links”:  
     - Nós com grau de entrada alto, mas conectados a usuários pouco influentes, tendem a ter PageRank menor do que nós que recebem poucas, mas “qualificadas” conexões;  
   - A presença de nós como 22933 e 8648 no top 10 com \(d = 0{,}99\) mostra isso: mesmo com poucos links de entrada, se esses links vêm de nós com PageRank alto, eles sobem muito no ranking.

3. **Papel do grau de saída**  
   - Um nó que “espalha” sua confiança em muitos outros (grau de saída alto) dilui seu impacto individual em cada destino;  
   - Já um nó com grau de saída 1 (como 22933 e 8648) concentra o PageRank recebido em um único destino, o que pode formar pequenos “ciclos” ou componentes que retêm massa de PageRank quando \(d\) é elevado.

---

## 7. Impacto da variação do fator de amortecimento \(d\)

### 7.1. \(d = 0{,}50\): teletransporte forte

- O teletransporte tem um peso maior (50%);  
- O grafo importa, mas a aleatoriedade ajuda a “democratizar” o ranking;  
- O top 10 ainda é dominado pelos grandes hubs de confiança (18, 737, etc.), mas as diferenças relativas são menores e a distribuição de PageRank tende a ser mais uniforme.

### 7.2. \(d = 0{,}85\): valor “padrão” da literatura

- Equilíbrio entre estrutura de links e teletransporte;  
- Ranking estável, com boa separação entre usuários muito influentes e o restante;  
- Convergência moderadamente rápida (31 iterações).

### 7.3. \(d = 0{,}99\): quase sem teletransporte

- A importância da estrutura de links fica extrema;  
- A convergência fica lenta (não convergiu em 100 iterações com `tol=1e-6`);  
- Aparecem comportamentos “estranhos”:
  - Nós com poucos links, mas conectados a clusters altamente conectados, sobem muito no ranking;  
  - A massa de PageRank tende a se concentrar em componentes fortemente conexas que funcionam como “buracos negros” de confiança.

Em termos práticos, isso mostra por que valores muito altos de \(d\) podem ser perigosos: o ranking fica mais sensível a detalhes da estrutura da rede e a pequenas irregularidades.

---

## 8. Limitações e possíveis extensões

**Limitações:**

- A análise é feita apenas sobre IDs numéricos; sem atributos dos usuários, não é possível rotular os nós (“especialistas em X”, “moderadores”, etc.);  
- Não foram calculadas outras medidas de centralidade (grau, betweenness, closeness) para comparação quantitativa com o PageRank.

**Possíveis extensões:**

- Comparar PageRank com centralidade de grau para mostrar casos em que a recursividade do PageRank “corrige” o ranking;  
- Analisar um subgrafo induzido pelos top 100 nós (densidade, modularidade);  
- Estudar cenários com limiares diferentes de tolerância e outro critério de parada (por exemplo, diferença no valor total de PageRank ou variação da ordenação do top-k).

---

## 9. Conclusão

Neste trabalho:

1. Foi implementado o algoritmo PageRank do zero, com tratamento de nós sem saída, critério de convergência por diferença máxima e estrutura de dados eficiente baseada em listas de adjacência.
2. A implementação foi validada contra `networkx.pagerank` para \(d = 0{,}85\), apresentando diferenças numéricas pequenas (diferença máxima da ordem de \(10^{-5}\)), o que indica correção do método.
3. Foram calculados os valores de PageRank da rede soc-Epinions1 para diferentes fatores de amortecimento (\(d = 0{,}50\), \(0{,}85\) e \(0{,}99\)), analisando convergência e o comportamento do ranking.
4. Identificou-se um conjunto estável de nós altamente influentes (como o nó 18 e outros hubs de alta entrada) e observou-se como o aumento de \(d\) tende a concentrar o PageRank em componentes específicas da rede, favorecendo alguns nós adicionais quando a aleatoriedade é reduzida.

Do ponto de vista de **aprendizado de máquina em grafos e análise de redes**, o exercício mostra claramente que:


 