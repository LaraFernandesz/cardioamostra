"""Calibração: curvas de 10 faixas e medidas de calibração por configuração.

Entrada: ``resultados/predicoes_comparacao.parquet`` (colunas
``experimento.COLUNAS_PREDICOES``), com a probabilidade fora da amostra de cada
pessoa na 1ª repetição da validação cruzada.

Medidas (as mesmas de van den Goorbergh et al., JAMIA 2022):

* **razão previsto/observado**: risco médio previsto ÷ proporção real de
  doentes. Acima de 1, o modelo exagera o risco.
* **intercepto de calibração** (calibração geral): o ``a`` em
  ``logit P(y=1) = a + logit(p)``. Zero é o ideal; negativo, risco superestimado.
* **inclinação de calibração**: o ``b`` em ``logit P(y=1) = a + b · logit(p)``.
  Um é o ideal; abaixo de 1, probabilidades extremas demais.

O Brier não é recalculado aqui: vem de ``comparacao.parquet`` (média das 15
rodadas), para que cada número apareça uma vez só no relatório.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FAIXAS = 10
CHAVES = ["base", "modelo", "estrategia"]
COLUNAS_CURVAS = [*CHAVES, "faixa", "inicio", "fim", "prevista", "observada", "pessoas"]
COLUNAS_MEDIDAS = [
    "previsto_medio",
    "observado",
    "razao_previsto_observado",
    "intercepto",
    "inclinacao",
]
METRICAS_TABELA = ["recall", "precisao", "f1", "pr_auc", "roc_auc", "brier"]

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Uma configuração
# --------------------------------------------------------------------------- #
def curva(y: Any, probabilidade: Any, faixas: int = FAIXAS) -> pd.DataFrame:
    """Curva de calibração com faixas iguais de probabilidade.

    A faixa ``i`` cobre ``[i/faixas, (i+1)/faixas)``; a última inclui o 1.
    Faixas sem ninguém ficam de fora.
    """
    y, p = _validar(y, probabilidade)
    bordas = np.linspace(0, 1, faixas + 1)
    faixa = np.clip(np.digitize(p, bordas[1:-1]), 0, faixas - 1)
    tabela = (
        pd.DataFrame({"faixa": faixa, "p": p, "y": y})
        .groupby("faixa")
        .agg(prevista=("p", "mean"), observada=("y", "mean"), pessoas=("y", "size"))
        .reset_index()
    )
    tabela["inicio"] = bordas[tabela["faixa"]]
    tabela["fim"] = bordas[tabela["faixa"] + 1]
    return tabela[["faixa", "inicio", "fim", "prevista", "observada", "pessoas"]]


def medidas(y: Any, probabilidade: Any) -> dict[str, float]:
    """Razão previsto/observado, intercepto e inclinação de calibração."""
    y, p = _validar(y, probabilidade)
    logito = _logito(p)
    um = np.ones((len(y), 1))
    intercepto = _logistica(um, y, deslocamento=logito)[0]
    inclinacao = _logistica(np.column_stack([um, logito]), y)[1]
    return {
        "previsto_medio": float(p.mean()),
        "observado": float(y.mean()),
        "razao_previsto_observado": float(p.mean() / y.mean()) if y.mean() > 0 else np.nan,
        "intercepto": float(intercepto),
        "inclinacao": float(inclinacao),
    }


# --------------------------------------------------------------------------- #
# Todas as configurações de um arquivo de predições
# --------------------------------------------------------------------------- #
def curvas(predicoes: pd.DataFrame, faixas: int = FAIXAS) -> pd.DataFrame:
    """Curva de cada base × modelo × estratégia, em formato longo."""
    partes = []
    for chave, grupo in predicoes.groupby(CHAVES, sort=False):
        tabela = curva(grupo["y"], grupo["probabilidade"], faixas)
        for nome, valor in zip(CHAVES, chave, strict=True):
            tabela[nome] = valor
        partes.append(tabela)
    if not partes:
        return pd.DataFrame(columns=COLUNAS_CURVAS)
    return pd.concat(partes, ignore_index=True)[COLUNAS_CURVAS]


def resumo(predicoes: pd.DataFrame) -> pd.DataFrame:
    """Medidas de calibração de cada base × modelo × estratégia."""
    linhas = []
    for chave, grupo in predicoes.groupby(CHAVES, sort=False):
        linha = dict(zip(CHAVES, chave, strict=True))
        linha.update(medidas(grupo["y"], grupo["probabilidade"]))
        linhas.append(linha)
    return pd.DataFrame(linhas, columns=[*CHAVES, *COLUNAS_MEDIDAS])


def tabela_final(comparacao: pd.DataFrame, calibracao: pd.DataFrame) -> pd.DataFrame:
    """Tabela final: média das 15 rodadas de cada métrica + medidas de calibração.

    ``comparacao`` está no formato longo de ``experimento.COLUNAS``;
    ``calibracao`` é a saída de :func:`resumo`.
    """
    metricas = (
        comparacao[comparacao["metrica"].isin(METRICAS_TABELA)]
        .pivot_table(index=CHAVES, columns="metrica", values="valor", aggfunc="mean")
        .reindex(columns=METRICAS_TABELA)
        .reset_index()
    )
    metricas.columns.name = None
    medidas_ = calibracao[[*CHAVES, "razao_previsto_observado", "intercepto", "inclinacao"]]
    return metricas.merge(medidas_, on=CHAVES, how="left")


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #
def _validar(y: Any, probabilidade: Any) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(probabilidade, dtype=float)
    if y_arr.shape != p_arr.shape or y_arr.ndim != 1:
        raise ValueError("y e probabilidade precisam ser vetores do mesmo tamanho.")
    if len(y_arr) == 0:
        raise ValueError("Não há predições para calcular a calibração.")
    if np.isnan(p_arr).any() or (p_arr < 0).any() or (p_arr > 1).any():
        raise ValueError("As probabilidades precisam estar entre 0 e 1.")
    if not set(np.unique(y_arr)) <= {0.0, 1.0}:
        raise ValueError("y precisa ser 0 ou 1.")
    return y_arr, p_arr


def _logito(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _logistica(
    X: np.ndarray, y: np.ndarray, deslocamento: np.ndarray | None = None, iteracoes: int = 100
) -> np.ndarray:
    """Regressão logística sem penalização por Newton-Raphson (só numpy).

    ``deslocamento`` entra no preditor linear com coeficiente fixo 1 (offset).
    Devolve ``nan`` se não convergir (por exemplo, separação perfeita).
    """
    offset = np.zeros(len(y)) if deslocamento is None else deslocamento
    beta = np.zeros(X.shape[1])
    for _ in range(iteracoes):
        q = 1 / (1 + np.exp(-(X @ beta + offset)))
        gradiente = X.T @ (y - q)
        hessiana = X.T @ (X * (q * (1 - q))[:, None])
        try:
            passo = np.linalg.solve(hessiana, gradiente)
        except np.linalg.LinAlgError:
            break
        beta = beta + passo
        if np.max(np.abs(passo)) < 1e-10:
            return beta
    return np.full(X.shape[1], np.nan)
