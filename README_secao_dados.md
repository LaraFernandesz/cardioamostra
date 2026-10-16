<!-- Seção do README escrita pela Pessoa 1 (Dados). Copiar para o README principal. -->

## Bases de dados

| Base | Pacientes | Com doença | Licença | Fonte |
|---|---|---|---|---|
| Personal Key Indicators of Heart Disease (BRFSS 2020) | 301.717 (após remover 18.078 duplicatas) | 9,04% (≈ 10:1) | [CC0 1.0 (domínio público)](https://creativecommons.org/publicdomain/zero/1.0/) | [Kaggle – K. Pytlak](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) |
| Heart Failure Clinical Records | 299 | 32,1% de óbitos | CC BY 4.0 | [UCI ML Repository, id 519](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) |
| Heart Disease – Cleveland | 303 | 45,9% (controle quase equilibrado) | CC BY 4.0 | [UCI ML Repository, id 45](https://archive.ics.uci.edu/dataset/45/heart+disease) |

**Citações**

- Pytlak, K. *Personal Key Indicators of Heart Disease*. Kaggle, 2022. Dados derivados do Behavioral Risk Factor Surveillance System (BRFSS) 2020, CDC.
- Chicco, D.; Jurman, G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20, 16, 2020. Base: *Heart Failure Clinical Records*, UCI Machine Learning Repository, 2020. https://doi.org/10.24432/C5Z89R
- Janosi, A.; Steinbrunn, W.; Pfisterer, M.; Detrano, R. *Heart Disease*. UCI Machine Learning Repository, 1988. https://doi.org/10.24432/C52P4X

**Como obter:** `python scripts/baixar_dados.py` (ou manualmente, ver `dados/brutos/LEIA-ME.md`). As bases não ficam no repositório.

### Preparação dos dados (`src/cardioamostra/dados.py`)

`carregar(base)` devolve `(X, y)`: `X` só numérico, `y` em 0/1.

- **BRFSS:** remoção das linhas duplicadas antes de qualquer codificação; `AgeCategory` (13 faixas) e `GenHealth` (Poor → Excellent) viram ordinais; variáveis Yes/No viram 0/1; `Race` e `Diabetic` recebem one-hot.
- **Insuficiência cardíaca:** alvo `DEATH_EVENT`. A coluna `time` (dias de acompanhamento) é **removida**: ela só é conhecida depois do desfecho e vazaria a resposta.
- **Cleveland:** alvo `num > 0` (qualquer grau de doença). Os 6 valores ausentes (`ca` 4, `thal` 2) são preenchidos com a moda; `cp`, `restecg`, `slope` e `thal` recebem one-hot.

Os testes em `tests/test_dados.py` conferem as contagens e a prevalência de cada base.

**Limitação:** o arquivo do Kaggle não tem os pesos amostrais do BRFSS, então os resultados valem para o modelo, não para a população dos EUA.
