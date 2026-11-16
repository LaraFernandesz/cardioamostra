"""Testes de calibracao.py com probabilidades simuladas de resposta conhecida."""

import numpy as np
import pandas as pd
import pytest

from cardioamostra.calibracao import (
    COLUNAS_CURVAS,
    COLUNAS_MEDIDAS,
    METRICAS_TABELA,
    curva,
    curvas,
    medidas,
    resumo,
    tabela_final,
)
from cardioamostra.experimento import COLUNAS


def _sigmoide(x):
    return 1 / (1 + np.exp(-x))


def _logito(p):
    return np.log(p / (1 - p))


@pytest.fixture(scope="module")
def calibrado():
    """Probabilidades verdadeiras: y é sorteado exatamente com probabilidade p."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.02, 0.98, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    return y, p


# ------------------------------------------------------------------ curva ----
def test_curva_conta_feita_a_mao():
    y = [0, 0, 1, 1]
    p = [0.05, 0.15, 0.15, 0.95]
    tabela = curva(y, p)
    assert tabela["faixa"].tolist() == [0, 1, 9]
    assert tabela["pessoas"].tolist() == [1, 2, 1]
    assert tabela["prevista"].tolist() == pytest.approx([0.05, 0.15, 0.95])
    assert tabela["observada"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert tabela["inicio"].tolist() == pytest.approx([0.0, 0.1, 0.9])
    assert tabela["fim"].tolist() == pytest.approx([0.1, 0.2, 1.0])


def test_curva_bordas_e_probabilidade_1():
    tabela = curva([0, 1, 1], [0.0, 0.1, 1.0])
    assert tabela["faixa"].tolist() == [0, 1, 9]


def test_curva_de_modelo_calibrado_fica_na_diagonal(calibrado):
    y, p = calibrado
    tabela = curva(y, p)
    assert len(tabela) == 10
    assert tabela["pessoas"].sum() == len(y)
    np.testing.assert_allclose(tabela["observada"], tabela["prevista"], atol=0.03)


# ---------------------------------------------------------------- medidas ----
def test_modelo_calibrado(calibrado):
    y, p = calibrado
    m = medidas(y, p)
    assert m["intercepto"] == pytest.approx(0, abs=0.05)
    assert m["inclinacao"] == pytest.approx(1, abs=0.05)
    assert m["razao_previsto_observado"] == pytest.approx(1, abs=0.02)


def test_risco_superestimado_da_intercepto_negativo(calibrado):
    """Como a reamostragem faz: soma 1,5 no logito de todo mundo."""
    y, p = calibrado
    m = medidas(y, _sigmoide(_logito(p) + 1.5))
    assert m["intercepto"] == pytest.approx(-1.5, abs=0.05)
    assert m["inclinacao"] == pytest.approx(1, abs=0.05)
    assert m["razao_previsto_observado"] > 1.2


def test_probabilidades_extremas_demais_dao_inclinacao_menor_que_1(calibrado):
    y, p = calibrado
    m = medidas(y, _sigmoide(2 * _logito(p)))
    assert m["inclinacao"] == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize(
    ("y", "p", "mensagem"),
    [
        ([0, 1], [0.2, 1.3], "entre 0 e 1"),
        ([0, 1], [0.2, np.nan], "entre 0 e 1"),
        ([0, 2], [0.2, 0.3], "0 ou 1"),
        ([0, 1], [0.2], "mesmo tamanho"),
        ([], [], "Não há"),
    ],
)
def test_entradas_invalidas_explicam_o_problema(y, p, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        medidas(y, p)


# ---------------------------------------------------- arquivo de predições ----
@pytest.fixture
def predicoes(calibrado):
    y, p = calibrado
    y, p = y[:2000], p[:2000]
    comum = {"base": "b", "modelo": "m", "y": y, "indice": np.arange(len(y))}
    return pd.concat(
        [
            pd.DataFrame({**comum, "estrategia": "nenhuma", "probabilidade": p}),
            pd.DataFrame(
                {**comum, "estrategia": "smote", "probabilidade": _sigmoide(_logito(p) + 1.5)}
            ),
        ],
        ignore_index=True,
    )


def test_curvas_e_resumo_tem_uma_entrada_por_configuracao(predicoes):
    c = curvas(predicoes)
    assert list(c.columns) == COLUNAS_CURVAS
    assert set(c["estrategia"]) == {"nenhuma", "smote"}
    assert c.groupby("estrategia")["pessoas"].sum().tolist() == [2000, 2000]

    r = resumo(predicoes).set_index("estrategia")
    assert list(r.columns) == ["base", "modelo", *COLUNAS_MEDIDAS]
    assert r.loc["smote", "intercepto"] < r.loc["nenhuma", "intercepto"]
    assert r.loc["smote", "razao_previsto_observado"] > r.loc["nenhuma", "razao_previsto_observado"]


def test_tabela_final_junta_metricas_e_calibracao(predicoes):
    linhas = []
    for estrategia, recall in [("nenhuma", 0.1), ("smote", 0.7)]:
        for rodada in range(2):
            for metrica in [*METRICAS_TABELA, "linhas_treino"]:
                valor = recall + rodada * 0.1 if metrica == "recall" else 0.5
                linhas.append(["b", "m", estrategia, rodada, metrica, valor])
    comparacao = pd.DataFrame(linhas, columns=COLUNAS)

    tabela = tabela_final(comparacao, resumo(predicoes)).set_index("estrategia")
    assert "linhas_treino" not in tabela.columns
    assert tabela.loc["smote", "recall"] == pytest.approx(0.75)
    assert tabela.loc["nenhuma", "recall"] == pytest.approx(0.15)
    assert {"razao_previsto_observado", "intercepto", "inclinacao"} <= set(tabela.columns)
    assert not tabela.isna().any().any()
