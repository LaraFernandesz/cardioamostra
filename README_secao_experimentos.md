<!-- Seção do README escrita pela Pessoa 2 (Experimentos). Copiar para o README principal. -->

## Protocolo experimental

Validação cruzada estratificada **5 × 3** (5 partes, 3 repetições, semente 42): cada configuração é
avaliada em 15 rodadas. A reamostragem acontece **só nas partes de treino**, dentro de um
`Pipeline` do imbalanced-learn (`escala → reamostragem → modelo`); a parte de teste tem apenas
pessoas reais. O teste `test_nenhuma_linha_sintetica_chega_ao_teste` falha se isso mudar.

### As 7 estratégias (`src/cardioamostra/estrategias.py`)

| Chave | Estratégia | O que faz |
|---|---|---|
| `nenhuma` | Nenhuma correção | Treina com os dados como estão, limiar 0,5 |
| `pesos` | Pesos de classe | `class_weight="balanced"`, nenhuma linha criada |
| `limiar` | Ajuste de limiar | Limiar escolhido no treino (validação interna 3 partes) para alcançar o **maior recall médio entre as 4 técnicas de reamostragem** da mesma base e modelo |
| `subamostragem` | Subamostragem aleatória | `RandomUnderSampler` |
| `sobreamostragem` | Sobreamostragem aleatória | `RandomOverSampler` |
| `smote` | SMOTE | `SMOTE(k_neighbors=5)`, exige pelo menos 6 doentes por parte de treino |
| `smote_tomek` | SMOTE + Tomek | `SMOTETomek` |

### Modelos (configurações fixas, sem busca de hiperparâmetros)

- `regressao_logistica`: `LogisticRegression(max_iter=2000)`
- `gradient_boosting`: `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False)`

### Como gerar os resultados

```bash
python scripts/rodar_experimentos.py --baseline                       # resultados/baseline.parquet
python scripts/rodar_experimentos.py --limite-linhas 50000            # resultados/comparacao.parquet
```

`--limite-linhas 50000` usa um subconjunto estratificado de 50.000 linhas nas bases maiores que
isso (só a BRFSS), como no artigo da IJACSA: o SMOTE + Tomek na base completa é lento demais.
O valor usado fica gravado nos metadados do Parquet.

### Contrato de dados

`comparacao.parquet` e `baseline.parquet`: formato longo, uma linha por
base × modelo × estratégia × rodada × métrica.

| Coluna | Valores |
|---|---|
| `base` | `brfss`, `insuficiencia`, `cleveland` |
| `modelo` | `regressao_logistica`, `gradient_boosting` |
| `estrategia` | as 7 chaves acima |
| `rodada` | 0 a 14 |
| `metrica` | `recall`, `precisao`, `f1`, `pr_auc`, `roc_auc`, `brier`, `linhas_treino`, `positivos_treino`, `linhas_apos`, `positivos_apos` |
| `valor` | número |

As 4 métricas de contagem alimentam a barra "Antes / Depois" da tela Laboratório.

`predicoes_comparacao.parquet`: probabilidades fora da amostra da 1ª repetição (cada pessoa
aparece uma vez por configuração), colunas `base, modelo, estrategia, indice, y, probabilidade`.
É a entrada das curvas de calibração (`calibracao.py`).

Os metadados de cada Parquet guardam data, versão do código (commit do git), semente, partes,
repetições, linhas por base e o recall-alvo usado no ajuste de limiar. Para ler:
`cardioamostra.experimento.ler_metadados(caminho)`.
