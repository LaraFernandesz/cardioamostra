"""Testes de apresentacao.py (a lógica do painel, sem Streamlit)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from cardioamostra.apresentacao import (
    TIPOS_PONTO,
    comparar_com_nenhuma,
    contagens,
    em_mil_pessoas,
    projetar,
    ranking,
    resumo_comparativo,
)
from cardioamostra.estrategias import ESTRATEGIAS, obter
from cardioamostra.experimento import COLUNAS


@pytest.fixture
def tabela():
    """Tabela final falsa: as técnicas sobem o recall e pioram o Brier; o limiar não."""
    linhas = []
    for estrategia in ESTRATEGIAS:
        corrige = estrategia not in ("nenhuma", "limiar")
        linhas.append(
            {
                "base": "b",
                "modelo": "m",
                "estrategia": estrategia,
                "recall": 0.1 if estrategia == "nenhuma" else 0.78,
                "precisao": 0.5 if estrategia == "nenhuma" else 0.23,
                "f1": 0.2,
                "pr_auc": 0.35,
                "roc_auc": 0.83,
                "brier": 0.17 if corrige else 0.07,
            }
        )
    return pd.DataFrame(linhas)


def test_ranking_por_brier_poe_o_menor_primeiro(tabela):
    r = ranking(tabela, "b", "m", "brier")
    assert list(r.index[:2]) == ["nenhuma", "limiar"]
    assert r["tecnica"].iloc[0] == "Nenhuma correção"


def test_ranking_por_recall_poe_o_maior_primeiro(tabela):
    r = ranking(tabela, "b", "m", "recall")
    assert r.index[-1] == "nenhuma"


def test_comparar_com_nenhuma(tabela):
    c = comparar_com_nenhuma(tabela, "b", "m", "smote")
    assert c["recall"]["valor"] == pytest.approx(0.78)
    assert c["recall"]["delta"] == pytest.approx(0.68)
    assert c["brier"]["delta"] == pytest.approx(0.10)
    assert comparar_com_nenhuma(tabela, "b", "m", "nenhuma")["recall"]["delta"] == 0


def test_resumo_comparativo_cita_os_numeros(tabela):
    texto = resumo_comparativo(tabela, "b", "m")
    assert "0,10" in texto and "0,78" in texto
    assert "0,170" in texto and "0,070" in texto
    assert "." not in texto.replace("probabilidades.", "").split("0,")[0]  # vírgula decimal


def test_contagens_media_das_rodadas():
    linhas = []
    for rodada, (antes, depois) in enumerate([(100, 180), (102, 182)]):
        for metrica, valor in [
            ("linhas_treino", antes),
            ("positivos_treino", 10),
            ("linhas_apos", depois),
            ("positivos_apos", 90),
        ]:
            linhas.append(["b", "m", "smote", rodada, metrica, valor])
    c = contagens(pd.DataFrame(linhas, columns=COLUNAS), "b", "m", "smote")
    assert c == {
        "linhas_antes": 101,
        "doentes_antes": 10,
        "linhas_depois": 181,
        "doentes_depois": 90,
    }


def test_em_mil_pessoas_soma_mil():
    m = em_mil_pessoas(prevalencia=0.09, recall=0.78, precisao=0.23)
    assert sum(m.values()) == pytest.approx(1000, abs=1)
    # 90 doentes; 0,78 × 90 = 70,2 encontrados; 70,2 / 0,23 − 70,2 = 235,0 alarmes falsos
    assert m["doentes_encontrados"] == 70
    assert m["doentes_perdidos"] == 20
    assert m["alarmes_falsos"] == 235


# ------------------------------------------------------------------- PCA ----
@pytest.fixture(scope="module")
def amostra():
    X, y = make_classification(
        n_samples=300, n_features=5, n_informative=3, weights=[0.85], random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


def test_sem_reamostragem_so_tem_pontos_reais(amostra):
    X, y = amostra
    p = projetar(X, y, obter("nenhuma"))
    assert len(p) == len(X)
    assert set(p["tipo"]) == {TIPOS_PONTO[0], TIPOS_PONTO[1]}
    assert (p["copias"] == 1).all()


def test_smote_marca_os_sinteticos(amostra):
    X, y = amostra
    p = projetar(X, y, obter("smote"))
    negativos = int((y == 0).sum())
    sinteticos = p[p["tipo"] == "Sintético"]
    assert len(sinteticos) == negativos - int(y.sum())
    assert (p["tipo"] == "Com doença (real)").sum() == int(y.sum())


def test_subamostragem_descarta_saudaveis(amostra):
    X, y = amostra
    p = projetar(X, y, obter("subamostragem"))
    assert (p["tipo"] == "Descartado").sum() == int((y == 0).sum() - y.sum())
    assert "Sintético" not in set(p["tipo"])


def test_sobreamostragem_repete_doentes(amostra):
    X, y = amostra
    p = projetar(X, y, obter("sobreamostragem"))
    doentes = p[p["tipo"] == "Com doença (real)"]
    assert doentes["copias"].sum() == int((y == 0).sum())
    assert "Sintético" not in set(p["tipo"])


def test_eixos_do_pca_nao_mudam_entre_tecnicas(amostra):
    X, y = amostra
    a = projetar(X, y, obter("nenhuma"))
    b = projetar(X, y, obter("smote")).iloc[: len(X)]
    np.testing.assert_allclose(a[["pc1", "pc2"]], b[["pc1", "pc2"]])
