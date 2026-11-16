"""Por que o SMOTE quase não muda o recall do gradient boosting na BRFSS?

    python scripts/investigar_smote_gb.py

Hipótese: 24 das 25 colunas da BRFSS só têm números inteiros (sim/não, faixa
etária, dias de saúde ruim, horas de sono). O SMOTE cria doentes com valores
impossíveis (7,3 dias, faixa etária 6,4), e as árvores aprendem a reconhecer o
sintético por esses valores, em vez de aprender o que é ser doente.

Teste: mesma validação cruzada estratificada de 5 partes (subconjunto de 50.000
pessoas), comparando o SMOTE puro com o SMOTE seguido de arredondar as colunas
inteiras. Grava ``resultados/investigacao_smote_gb.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

LIMITE = 50_000
SEMENTE = 42


def main() -> int:
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import recall_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from cardioamostra.dados import carregar
    from cardioamostra.experimento import criar_classificador, subamostrar_estratificado

    X_df, y_s = subamostrar_estratificado(*carregar("brfss"), LIMITE)
    X, y = X_df.to_numpy(), y_s.to_numpy()
    inteiras = [j for j in range(X.shape[1]) if np.all(np.mod(X[:, j], 1) == 0)]
    print(f"{len(inteiras)} de {X.shape[1]} colunas só têm números inteiros")

    def smote(Xt, yt, arredondar):
        Xr, yr = SMOTE(k_neighbors=5, random_state=SEMENTE).fit_resample(Xt, yt)
        if arredondar:
            Xr[:, inteiras] = np.round(Xr[:, inteiras])
        return Xr, yr

    variantes = {
        "nenhuma": lambda Xt, yt: (Xt, yt),
        "sobreamostragem": lambda Xt, yt: RandomOverSampler(random_state=SEMENTE).fit_resample(
            Xt, yt
        ),
        "smote": lambda Xt, yt: smote(Xt, yt, arredondar=False),
        "smote_arredondado": lambda Xt, yt: smote(Xt, yt, arredondar=True),
    }
    modelos = {
        "gradient_boosting": lambda: criar_classificador("gradient_boosting", SEMENTE, False),
        "regressao_logistica": lambda: LogisticRegression(max_iter=2000),
    }

    linhas = []
    divisor = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    for parte, (treino, teste) in enumerate(divisor.split(X, y)):
        for nome, reamostrar in variantes.items():
            Xr, yr = reamostrar(X[treino], y[treino])
            sinteticos = Xr[len(treino) :] if nome.startswith("smote") else None
            escala = StandardScaler().fit(Xr)
            for modelo, criar in modelos.items():
                m = criar().fit(escala.transform(Xr), yr)
                prob = m.predict_proba(escala.transform(X[teste]))[:, 1]
                linha = {
                    "parte": parte,
                    "variante": nome,
                    "modelo": modelo,
                    "recall": recall_score(y[teste], (prob >= 0.5).astype(int)),
                    "prob_doentes_reais_teste": prob[y[teste] == 1].mean(),
                    "prob_sinteticos_treino": np.nan,
                    "sinteticos_com_valor_impossivel": np.nan,
                }
                if sinteticos is not None:
                    quebrado = np.mod(sinteticos[:, inteiras], 1) != 0
                    linha["sinteticos_com_valor_impossivel"] = quebrado.any(axis=1).mean()
                    linha["prob_sinteticos_treino"] = m.predict_proba(
                        escala.transform(sinteticos)
                    )[:, 1].mean()
                linhas.append(linha)
        print(f"  parte {parte + 1}/5 pronta")

    tabela = (
        pd.DataFrame(linhas)
        .groupby(["modelo", "variante"], sort=False)
        .mean(numeric_only=True)
        .drop(columns="parte")
        .round(3)
    )
    tabela.to_csv(RAIZ / "resultados" / "investigacao_smote_gb.csv", sep=";", decimal=",")
    pd.set_option("display.width", 160)
    print(tabela.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
