"""Protocolo experimental: validação cruzada estratificada 5 × 3 e métricas.

A reamostragem fica sempre dentro de um ``Pipeline`` do imbalanced-learn, que só
reamostra no ``fit`` (partes de treino). Na previsão, o amostrador é ignorado e
o modelo vê apenas pessoas reais da parte de teste.

Formato dos resultados (contrato de dados com análise e painel): tabela longa
com as colunas ``COLUNAS``, uma linha por base × modelo × estratégia × rodada ×
métrica.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from cardioamostra.estrategias import Estrategia, de_reamostragem

SEMENTE = 42
PARTES = 5
REPETICOES = 3

COLUNAS = ["base", "modelo", "estrategia", "rodada", "metrica", "valor"]
COLUNAS_PREDICOES = ["base", "modelo", "estrategia", "indice", "y", "probabilidade"]

METRICAS_DESEMPENHO = ["recall", "precisao", "f1", "pr_auc", "roc_auc", "brier"]
METRICAS_CONTAGEM = ["linhas_treino", "positivos_treino", "linhas_apos", "positivos_apos"]

MODELOS: dict[str, str] = {
    "regressao_logistica": "Regressão logística",
    "gradient_boosting": "Gradient boosting",
}

CHAVE_METADADOS = b"cardioamostra"


def criar_classificador(chave: str, semente: int, pesos_de_classe: bool) -> BaseEstimator:
    """Os 2 modelos do projeto, com configurações fixas e documentadas."""
    pesos = "balanced" if pesos_de_classe else None
    if chave == "regressao_logistica":
        return LogisticRegression(max_iter=2000, class_weight=pesos, random_state=semente)
    if chave == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            early_stopping=False,
            class_weight=pesos,
            random_state=semente,
        )
    raise KeyError(f"Modelo '{chave}' não existe. Use uma de: {', '.join(MODELOS)}")


class ContadorDeLinhas(TransformerMixin, BaseEstimator):
    """Passo neutro que anota quantas linhas chegaram ao modelo no ``fit``.

    Fica logo depois do amostrador, então registra o tamanho do treino já
    reamostrado (a barra "Depois" da tela Laboratório).
    """

    def fit(self, X: Any, y: Any = None) -> ContadorDeLinhas:
        self.linhas_ = int(len(X))
        self.positivos_ = int(np.sum(y)) if y is not None else 0
        return self

    def transform(self, X: Any) -> Any:
        return X


def escolher_limiar(y: Any, probabilidade: Any, recall_alvo: float) -> float:
    """Maior limiar que ainda encontra pelo menos ``recall_alvo`` dos doentes."""
    y = np.asarray(y)
    probabilidade = np.asarray(probabilidade)
    if not 0 < recall_alvo <= 1:
        raise ValueError("recall_alvo precisa estar entre 0 (exclusivo) e 1.")
    positivos = np.sort(probabilidade[y == 1])[::-1]
    if len(positivos) == 0:
        raise ValueError("Não há positivos para calibrar o limiar.")
    quantos = int(np.ceil(recall_alvo * len(positivos) - 1e-9))
    quantos = min(max(quantos, 1), len(positivos))
    return float(positivos[quantos - 1])


class ClassificadorComLimiar(ClassifierMixin, BaseEstimator):
    """Envolve um estimador e troca o limiar 0,5 por um escolhido só no treino.

    O limiar sai de previsões fora da amostra (validação cruzada interna sobre a
    parte de treino), nunca da parte de teste. ``predict_proba`` não muda: é por
    isso que o ajuste de limiar preserva a calibração.
    """

    def __init__(
        self,
        estimador: Any = None,
        recall_alvo: float = 0.75,
        partes_internas: int = 3,
        semente: int = SEMENTE,
    ) -> None:
        self.estimador = estimador
        self.recall_alvo = recall_alvo
        self.partes_internas = partes_internas
        self.semente = semente

    def fit(self, X: Any, y: Any) -> ClassificadorComLimiar:
        y = np.asarray(y)
        divisor = StratifiedKFold(
            n_splits=self.partes_internas, shuffle=True, random_state=self.semente
        )
        probabilidade = cross_val_predict(
            clone(self.estimador), X, y, cv=divisor, method="predict_proba"
        )[:, 1]
        self.limiar_ = escolher_limiar(y, probabilidade, self.recall_alvo)
        self.estimador_ = clone(self.estimador).fit(X, y)
        self.classes_ = self.estimador_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimador_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.limiar_).astype(int)


def montar_pipeline(
    estrategia: Estrategia,
    modelo: str,
    semente: int = SEMENTE,
    recall_alvo: float | None = None,
) -> BaseEstimator:
    """Escala → (reamostragem) → contador → modelo, tudo dentro de um Pipeline.

    A escala vem antes do SMOTE porque ele mede distância entre vizinhos.
    """
    passos: list[tuple[str, Any]] = [("escala", StandardScaler())]
    if estrategia.criar_amostrador is not None:
        passos.append(("amostragem", estrategia.criar_amostrador(semente)))
    passos.append(("contador", ContadorDeLinhas()))
    passos.append(("modelo", criar_classificador(modelo, semente, estrategia.pesos_de_classe)))
    pipeline = Pipeline(passos)

    if not estrategia.ajusta_limiar:
        return pipeline
    if recall_alvo is None:
        raise ValueError(
            "A estratégia de ajuste de limiar precisa de recall_alvo "
            "(o recall da melhor técnica de reamostragem)."
        )
    return ClassificadorComLimiar(pipeline, recall_alvo=recall_alvo, semente=semente)


def calcular_metricas(y: Any, probabilidade: Any, previsao: Any) -> dict[str, float]:
    """As 6 métricas do projeto para uma parte de teste."""
    y = np.asarray(y)
    return {
        "recall": float(recall_score(y, previsao, zero_division=0)),
        "precisao": float(precision_score(y, previsao, zero_division=0)),
        "f1": float(f1_score(y, previsao, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probabilidade)),
        "roc_auc": float(roc_auc_score(y, probabilidade)),
        "brier": float(brier_score_loss(y, probabilidade)),
    }


def _contador(estimador: Any) -> ContadorDeLinhas | None:
    alvo = getattr(estimador, "estimador_", estimador)
    passos = getattr(alvo, "named_steps", {})
    return passos.get("contador")


def avaliar(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    estrategia: Estrategia,
    modelo: str,
    base: str,
    semente: int = SEMENTE,
    partes: int = PARTES,
    repeticoes: int = REPETICOES,
    recall_alvo: float | None = None,
    criar_pipeline: Callable[..., BaseEstimator] = montar_pipeline,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda a validação cruzada repetida de uma configuração.

    Devolve ``(resultados, predicoes)``: os resultados no formato longo de
    ``COLUNAS`` e as probabilidades fora da amostra da 1ª repetição (cada
    pessoa aparece uma vez), usadas nas curvas de calibração.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).astype(int)
    divisor = RepeatedStratifiedKFold(n_splits=partes, n_repeats=repeticoes, random_state=semente)

    linhas: list[dict[str, Any]] = []
    predicoes: list[pd.DataFrame] = []
    for rodada, (treino, teste) in enumerate(divisor.split(X_arr, y_arr)):
        estimador = criar_pipeline(estrategia, modelo, semente=semente, recall_alvo=recall_alvo)
        estimador.fit(X_arr[treino], y_arr[treino])

        probabilidade = estimador.predict_proba(X_arr[teste])[:, 1]
        previsao = estimador.predict(X_arr[teste])
        valores = calcular_metricas(y_arr[teste], probabilidade, previsao)

        valores["linhas_treino"] = float(len(treino))
        valores["positivos_treino"] = float(y_arr[treino].sum())
        contador = _contador(estimador)
        if contador is not None:
            valores["linhas_apos"] = float(contador.linhas_)
            valores["positivos_apos"] = float(contador.positivos_)

        for metrica, valor in valores.items():
            linhas.append(
                {
                    "base": base,
                    "modelo": modelo,
                    "estrategia": estrategia.chave,
                    "rodada": rodada,
                    "metrica": metrica,
                    "valor": valor,
                }
            )

        if rodada < partes:
            predicoes.append(
                pd.DataFrame(
                    {
                        "base": base,
                        "modelo": modelo,
                        "estrategia": estrategia.chave,
                        "indice": teste,
                        "y": y_arr[teste].astype("int8"),
                        "probabilidade": probabilidade.astype("float32"),
                    }
                )
            )

    resultados = pd.DataFrame(linhas, columns=COLUNAS)
    return resultados, pd.concat(predicoes, ignore_index=True)[COLUNAS_PREDICOES]


def media(resultados: pd.DataFrame, metrica: str) -> pd.Series:
    """Média de uma métrica nas rodadas, por base × modelo × estratégia."""
    filtro = resultados[resultados["metrica"] == metrica]
    return filtro.groupby(["base", "modelo", "estrategia"])["valor"].mean()


def recall_alvo_para_limiar(resultados: pd.DataFrame, base: str, modelo: str) -> float:
    """Maior recall médio entre as técnicas de reamostragem, para a mesma base e modelo.

    É o recall que o ajuste de limiar precisa alcançar para a comparação ser
    em pé de igualdade.
    """
    chaves = {e.chave for e in de_reamostragem()}
    recalls = media(resultados, "recall")
    recalls = recalls[
        (recalls.index.get_level_values("base") == base)
        & (recalls.index.get_level_values("modelo") == modelo)
        & (recalls.index.get_level_values("estrategia").isin(chaves))
    ]
    if recalls.empty:
        raise ValueError(
            f"Nenhuma técnica de reamostragem rodou para {base}/{modelo}; "
            "rode-as antes do ajuste de limiar."
        )
    return float(recalls.max())


def subamostrar_estratificado(
    X: pd.DataFrame, y: pd.Series, limite: int, semente: int = SEMENTE
) -> tuple[pd.DataFrame, pd.Series]:
    """Mantém ``limite`` linhas preservando a proporção de doentes."""
    if len(X) <= limite:
        return X, y
    proporcao = limite / len(X)
    escolhidos = np.sort(
        pd.Series(np.arange(len(y)))
        .groupby(y.to_numpy())
        .sample(frac=proporcao, random_state=semente)
        .to_numpy()
    )
    return X.iloc[escolhidos].reset_index(drop=True), y.iloc[escolhidos].reset_index(drop=True)


def salvar_parquet(tabela: pd.DataFrame, caminho: Path, metadados: dict[str, Any]) -> None:
    """Grava o Parquet com data, versão do código e parâmetros nos metadados."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    arrow = pa.Table.from_pandas(tabela, preserve_index=False)
    existentes = dict(arrow.schema.metadata or {})
    existentes[CHAVE_METADADOS] = json.dumps(metadados, ensure_ascii=False).encode("utf-8")
    pq.write_table(arrow.replace_schema_metadata(existentes), caminho)


def ler_metadados(caminho: Path) -> dict[str, Any]:
    """Lê de volta os metadados gravados por ``salvar_parquet``."""
    bruto = (pq.read_schema(caminho).metadata or {}).get(CHAVE_METADADOS)
    return json.loads(bruto.decode("utf-8")) if bruto else {}


def metadados_base(versao: str, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    dados: dict[str, Any] = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_codigo": versao,
        "semente": SEMENTE,
        "partes": PARTES,
        "repeticoes": REPETICOES,
    }
    dados.update(extras or {})
    return dados


def concatenar(tabelas: Iterable[pd.DataFrame]) -> pd.DataFrame:
    lista = [t for t in tabelas if not t.empty]
    return pd.concat(lista, ignore_index=True) if lista else pd.DataFrame(columns=COLUNAS)
