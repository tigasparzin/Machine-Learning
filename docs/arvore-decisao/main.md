##	Exploração dos Dados

A base possui 140 registros e 9 colunas.

*Estágio acadêmico*: Referente a qual estagio academico o respondente está. Sendo 71% estudantes de graduação, 21% no ensino médio e 8% de pós graduação.

*Pressão dos colegas*: O quanto o aluno se sente pressionado pelos colegas numa escala de 1 a 5, com média das respostas de 3,07.

*Pressão acadêmica da família*: O quanto o aluno se sente pressionado pela familia numa escala de 1 a 5, com média das respostas de 3,17.

*Ambiente de estudo*: Como o aluno classifica o ambiente onde estuda entre as categorias: Peaceful (49% das respostas), Noisy(24% das respostas) e Disrupted(27% das respostas).

*Estratégia de enfrentamento*: Como o aluno enfrenta o estresse, entre as opções: Analisar a situacao com inteligencia (62%), Colapso emocional (23%), apoio de amigo/familia (15%).

*Maus habitos*: Diz se o respondente tem maus habitos, como fumar ou beber, entre as opções: "Não" (88%), "Sim" (7%) e 5% preferiram não responder.

*Competição acadêmica*: Classificar a competicao academica na sua vida, em uma escala escala de 1 a 5, tendo média das respostas de 3,49.

*Índice de estresse*: Classificar o nivel de estresse academico, em uma escala de 1 a 5, tendo média de 3,72.

---

##	Pré-processamento

#### Remoção de colunas irrelevantes
A coluna Timestamp foi removida por não agregar informações para a previsão.

#### Variável-alvo
A variável alvo definida foi "Rate your academic stress index".

#### Tratamento de missing value
A base apresenta apenas um valor nulo em Study Environment, que será preenchido com "Peaceful", valor mais frequente.

#### Codificação de variáveis categóricas
As colunas "Your Academic Stage", "Study Environment", "What coping strategy you use as a student?" e "Do you have any bad habits?", são categóricas.

Foram convertidas em valores numéricos utilizando Label Encoding, para que a árvore de decisão consiga interpretá-las.

#### Features e target

feature (X): estágio acadêmico, pressão dos colegas, pressão da família, ambiente de estudo, estratégia de enfrentamento, hábitos nocivos, nível de competição acadêmica.

target (y): índice de estresse acadêmico (1 a 5).


##	Divisão dos Dados

70% dos registros da base serão separados para treino e 30% separados para testes de acuracia

##	Treinamento do Modelo

```python exec="1" html="1"
--8<-- "docs/arvore-decisao/student_decision_tree.py"
```

##	Avaliação do Modelo


##  Conclusão