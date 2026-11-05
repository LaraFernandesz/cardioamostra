import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from cardioamostra.estrategias import obter
from cardioamostra.experimento import (
    COLUNAS,
    COLUNAS_PREDICOES,
    METRICAS_CONTAGEM,
    METRICAS_DESEMPENHO,
    ClassificadorComLimiar,
    avaliar,
    calcular_metricas,
    escolher_limiar,
    ler_metadados,
    montar_pipeline,
    recall_alvo_para_limiar,
    salvar_parquet,
    subamostrar_estratificado,
)

REGISTRO: dict[str, list[np.ndarray]] = {"fit": [], "previsao": []}


class Espiao(ClassifierMixin, BaseEstimator):
    """Classificador falso que guarda tudo o que recebe."""

    def fit(self, X, y):
        REGISTRO["fit"].append(np.asarray(X).copy())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        REGISTRO["previsao"].append(np.asarray(X).copy())
        return np.tile([0.5, 0.5], (len(X), 1))

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def pipeline_espiao(estrategia, modelo, semente=42, recall_alvo=None):
    return Pipeline(
        [("amostragem", SMOTE(k_neighbors=5, random_state=semente)), ("modelo", Espiao())]
    )


@pytest.fixture(scope="module")
def dados():
    X, y = make_classification(
        n_samples=600, n_features=5, n_informative=3, weights=[0.85], random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


def linhas_como_conjunto(matriz):
    return {tuple(np.round(linha, 10)) for linha in matriz}


def test_nenhuma_linha_sintetica_chega_ao_teste(dados):
    X, y = dados
    REGISTRO["fit"].clear()
    REGISTRO["previsao"].clear()

    avaliar(X, y, obter("smote"), "espiao", "teste", criar_pipeline=pipeline_espiao)

    originais = linhas_como_conjunto(X.to_numpy())
    assert len(REGISTRO["previsao"]) == 15
    assert sum(len(p) for p in REGISTRO["previsao"]) == 3 * len(X)
    for vistas in REGISTRO["previsao"]:
        assert linhas_como_conjunto(vistas) <= originais

    sinteticas = linhas_como_conjunto(REGISTRO["fit"][0]) - originais
    assert len(sinteticas) > 0
    assert len(REGISTRO["fit"][0]) > len(X) * 4 / 5


def test_partes_de_teste_cobrem_todas_as_pessoas_sem_repetir(dados):
    X, y = dados
    REGISTRO["previsao"].clear()
    avaliar(X, y, obter("smote"), "espiao", "teste", criar_pipeline=pipeline_espiao)
    primeira_repeticao = np.vstack(REGISTRO["previsao"][:5])
    assert len(linhas_como_conjunto(primeira_repeticao)) == len(X)


def test_metricas_conferem_com_conta_feita_a_mao():
    y = [1, 1, 0, 0]
    probabilidade = [0.9, 0.4, 0.6, 0.1]
    previsao = [1, 0, 1, 0]
    metricas = calcular_metricas(y, probabilidade, previsao)
    assert metricas["recall"] == pytest.approx(0.5)
    assert metricas["precisao"] == pytest.approx(0.5)
    assert metricas["f1"] == pytest.approx(0.5)
    assert metricas["roc_auc"] == pytest.approx(0.75)
    assert metricas["brier"] == pytest.approx(0.185)
    assert metricas["pr_auc"] == pytest.approx(5 / 6)


def test_escolher_limiar_conta_a_mao():
    y = [1, 1, 1, 1, 0]
    probabilidade = [0.9, 0.8, 0.3, 0.2, 0.5]
    assert escolher_limiar(y, probabilidade, 0.5) == pytest.approx(0.8)
    assert escolher_limiar(y, probabilidade, 0.75) == pytest.approx(0.3)
    assert escolher_limiar(y, probabilidade, 1.0) == pytest.approx(0.2)


def test_limiar_nao_muda_as_probabilidades(dados):
    X, y = dados
    sem = montar_pipeline(obter("nenhuma"), "regressao_logistica").fit(X, y)
    com = montar_pipeline(obter("limiar"), "regressao_logistica", recall_alvo=0.8).fit(X, y)
    assert isinstance(com, ClassificadorComLimiar)
    np.testing.assert_allclose(sem.predict_proba(X), com.predict_proba(X))
    assert com.limiar_ < 0.5


def test_limiar_alcanca_recall_parecido_com_o_alvo(dados):
    X, y = dados
    tabela, _ = avaliar(X, y, obter("limiar"), "regressao_logistica", "teste", recall_alvo=0.8)
    recall = tabela.loc[tabela["metrica"] == "recall", "valor"].mean()
    assert recall == pytest.approx(0.8, abs=0.1)


def test_formato_do_contrato_de_dados(dados):
    X, y = dados
    tabela, predicoes = avaliar(X, y, obter("smote"), "gradient_boosting", "teste")
    assert list(tabela.columns) == COLUNAS
    assert list(predicoes.columns) == COLUNAS_PREDICOES
    assert sorted(tabela["rodada"].unique()) == list(range(15))
    assert set(tabela["metrica"]) == set(METRICAS_DESEMPENHO + METRICAS_CONTAGEM)
    assert not tabela.isna().any().any()
    assert len(predicoes) == len(X)
    assert sorted(predicoes["indice"]) == list(range(len(X)))


def test_contagens_mostram_o_treino_antes_e_depois(dados):
    X, y = dados
    tabela, _ = avaliar(X, y, obter("smote"), "regressao_logistica", "teste")
    medias = tabela.groupby("metrica")["valor"].mean()
    assert medias["linhas_treino"] == pytest.approx(len(X) * 4 / 5, abs=1)
    assert medias["linhas_apos"] > medias["linhas_treino"]
    assert medias["positivos_apos"] * 2 == pytest.approx(medias["linhas_apos"])


def test_recall_alvo_e_o_maior_entre_as_reamostragens():
    linhas = []
    for estrategia, recall in [("nenhuma", 0.9), ("smote", 0.6), ("subamostragem", 0.7)]:
        for rodada in range(2):
            linhas.append(["b", "m", estrategia, rodada, "recall", recall])
    tabela = pd.DataFrame(linhas, columns=COLUNAS)
    assert recall_alvo_para_limiar(tabela, "b", "m") == pytest.approx(0.7)


def test_recall_alvo_sem_reamostragem_falha():
    tabela = pd.DataFrame([["b", "m", "nenhuma", 0, "recall", 0.1]], columns=COLUNAS)
    with pytest.raises(ValueError, match="reamostragem"):
        recall_alvo_para_limiar(tabela, "b", "m")


def test_subamostra_mantem_a_prevalencia(dados):
    X, y = dados
    X_menor, y_menor = subamostrar_estratificado(X, y, 300)
    assert len(X_menor) == pytest.approx(300, abs=2)
    assert y_menor.mean() == pytest.approx(y.mean(), abs=0.01)
    assert subamostrar_estratificado(X, y, 10_000)[0] is X


def test_parquet_guarda_os_metadados(tmp_path):
    tabela = pd.DataFrame([["b", "m", "nenhuma", 0, "recall", 0.5]], columns=COLUNAS)
    caminho = tmp_path / "r.parquet"
    salvar_parquet(tabela, caminho, {"versao_codigo": "abc123", "semente": 42})
    assert ler_metadados(caminho) == {"versao_codigo": "abc123", "semente": 42}
    pd.testing.assert_frame_equal(pd.read_parquet(caminho), tabela)
