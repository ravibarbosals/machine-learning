# Resultados – Classificador de vacinas

Silver: 4716 linhas | Linha de base (vacina mais comum): 42,8%

## Aula 10 – Melhorando o classificador

| # | O que mudou | Acurácia (validação) | Acurácia (teste) |
|---|-------------|----------------------|------------------|
| 0 | Modelo original (max_depth=4, idade/dose/ano/mes + sexo/raca_cor/distrito) | 64,2% | 63,2% |
| 1 | Troca `ano`+`mes` por `dia_ano` (dia do ano, 1–365) | 72,0% | 70,7% |
| 2 | **GridSearchCV (5 folds) em max_depth, min_samples_leaf, criterion → entropy, depth 12, leaf 10** | **81,3%** | **81,4%** |
| 3 | + coluna `unidade_saude` | 81,3% | 81,4% |
| 4 | + `unidade_saude`, `grupo`, `categoria` | 81,2% | 80,9% |
| 5 | Árvore sem limite (max_depth=None) — só para comparação | 77,8% | 75,7% |
| 6 | GridSearchCV com `ano`+`mes` (sem `dia_ano`) — prova de que o ganho depende da data fina | 69,0% | 68,0% |

**Modelo final: tentativa #2.** As colunas novas (#3 e #4) não melhoraram a validação; com empate,
mantive o modelo mais simples. As tentativas #5 e #6 foram feitas só para responder às perguntas 3 e 4.

### Perguntas para discussão

**1. Com a coluna `lote` o modelo passa de 90%. Por que isso não é um bom modelo?**
Porque é *data leakage* (vazamento de dados): cada lote pertence a um único fabricante, então
o lote praticamente já *é* a resposta. O modelo aprenderia uma tabela "lote → vacina" e não o
perfil de quem tomou cada vacina. Na prática, quem tem o lote em mãos já sabe a vacina; o
modelo não teria utilidade e os 90% seriam uma ilusão.

**2. Acurácia de 80% significa que o modelo é bom para todas as vacinas?**
Não. No modelo final, a acurácia é 81%, mas o recall da CORONAVAC pediátrica é 36% e o da
CORONAVAC adulta é 50%. Classes pequenas (25 e 60 exemplos no teste) pesam pouco na
acurácia geral. O `macro avg` do f1 (0,73), que dá peso igual a cada vacina, mostra melhor isso.

**3. Uma árvore sem limite acerta quase 100% no treino. Por que não se repete no teste?**
Porque ela decora o treino: cria folhas para casos individuais, inclusive ruído. No nosso
experimento #5: 99,4% no treino, mas 77,8% na validação e 75,7% no teste. Limitando
`max_depth=12` e `min_samples_leaf=10`, o treino cai para 84,8%, mas o teste sobe para 81,4%:
a árvore passa a aprender padrões gerais.

**4. Qual mudança teve maior impacto? Por quê?**
A troca de `mes` por `dia_ano`. Sozinha deu +7,8 pontos na validação (64,2% → 72,0%), e foi
ela que permitiu o ganho do GridSearchCV: as campanhas trocavam de vacina em dias específicos,
e o mês juntava campanhas diferentes no mesmo valor. Sem uma data fina, profundidade extra
não tem o que aprender.
Prova (experimento #6): o mesmo GridSearchCV usando `mes` chega só a 69,0% na validação.
Grid sozinho ≈ +5 pontos, `dia_ano` sozinho ≈ +8, e os dois juntos ≈ +17.

## Aula 11 – Previsões na camada gold

O `train.py` treina o modelo final (#2) e o grava com `salvar()` em `modelos/classificador.joblib`.
O `prever.py` só carrega o modelo com `carregar()`, sem treinar de novo.

**Conferência:** o modelo carregado, avaliado no mesmo `X_teste`, deu **81,4%**, a mesma acurácia
(e o mesmo relatório por vacina) do modelo recém-treinado.

### Perfis novos (`perfis_novos.csv` → `gold_previsoes`)

| Perfil | Idade | Dose | Data | Distrito | Vacina prevista | Confiança |
|--------|-------|------|------|----------|-----------------|-----------|
| `crianca` | 8 | 1 | 2022-02-15 | DS 4 | COMIRNATY (PFIZER) - PEDIÁTRICA | 1,000 |
| `reforco` | 45 | 3 | 2022-06-20 | DS 2 | JANSSEN COVID-19 VACCINE (JOHNSON & JOHNSON) | 0,667 |
| `adulto_primeira_dose` | 25 | 1 | 2022-01-20 | DS 3 | CHADOX1NCOV-19 - OXFORD/ASTRAZENECA (FIOCRUZ) | 0,474 |
| `idoso_quarta_dose` | 80 | 4 | 2022-09-05 | DS 1 | COMIRNATY (PFIZER) | 0,562 |
| `distrito_inexistente` | 30 | 1 | 2022-02-10 | **DS 99** | CHADOX1NCOV-19 - OXFORD/ASTRAZENECA (FIOCRUZ) | **1,000** |
| `fora_do_periodo` | 60 | 4 | **2023-08-15** | DS 6 | CHADOX1NCOV-19 - OXFORD/ASTRAZENECA (FIOCRUZ) | **0,818** |

A **confiança** é a maior probabilidade do `predict_proba`: na árvore, é a proporção da vacina
prevista entre as pessoas do treino que caíram na mesma folha (cada folha tem pelo menos 10 pessoas).

- `crianca`: o caso mais fácil; o primeiro corte da árvore é `idade <= 11.5`.
- `adulto_primeira_dose`: confiança abaixo de 50%. A folha mistura vacinas: é a mesma confusão
  AstraZeneca ↔ Pfizer ↔ Janssen que aparece na matriz de confusão.

### Análise: `distrito_inexistente` (DS 99) – confiança 1,0 num distrito que não existe

No `preparar_entrada`, o `reindex(columns=self.colunas, fill_value=0)` descarta a coluna
`distrito_sanitario_DS 99` (o treino nunca a viu) e deixa todas as colunas `DS 1` a `DS 8` com 0.
A árvore não tem a opção "não sei": ela só pergunta coisas como "é DS 4?", recebe "não" e segue.

Testando o mesmo perfil em todos os distritos, DS 1, 2, 3, 5, 6, 7, 8 e DS 99 dão AstraZeneca
com confiança 1,0; só o DS 4 muda para Pfizer. Ou seja, o DS 99 é tratado como "qualquer distrito
que não é o DS 4". A confiança 1,0 não mede o quanto o perfil é confiável: mede só a pureza da
folha onde ele caiu. **Confiança alta não significa previsão confiável.**

### Análise: `fora_do_periodo` (2023) – o `dia_ano` joga fora o ano

A feature `dia_ano` guarda só o dia do ano (1 a 365). 15/08/2023 vira 227, o mesmo valor de
15/08/2022. Testando o mesmo perfil em 2022, 2023 e 2030, as três previsões são idênticas
(AstraZeneca, 0,818). Para 2022 a resposta é coerente com os dados: 44 das 54 pessoas com 60+
anos e 4ª dose entre julho e setembro de 2022 tomaram AstraZeneca. Para 2023, o modelo não tem
nenhum dado; ele só repete 2022.

Mesmo usando o ano como feature, isso não se resolveria: a silver só tem 2022, e uma árvore não
extrapola, então qualquer ano depois de 2022 cairia no mesmo ramo.

**Conclusão:** o modelo só sabe responder sobre o mundo que viu no treino (vacinações de 2022 nos
distritos DS 1 a DS 8). Fora desse domínio ele não avisa nem baixa a confiança: simplesmente
responde. Quem usa o modelo precisa conhecer esse limite, ou o código precisa validar a entrada
antes de prever.
