"""Fidelidade dos dados da interface e falhas recuperáveis."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cardioamostra.apresentacao import contagens
from cardioamostra.estrategias import ESTRATEGIAS
from cardioamostra.interface import (
    METRICAS,
    ErroResultados,
    carregar_interface,
    ler_tabela,
    projecoes_da_amostra,
)

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"


@pytest.fixture(scope="module")
def dados():
    if not (RESULTADOS / "tabela_final.parquet").exists():
        pytest.skip("resultados não gerados")
    return carregar_interface(RESULTADOS)


def test_todas_as_configuracoes_preservam_metricas_e_contagens(dados):
    tabela = pd.read_parquet(RESULTADOS / "tabela_final.parquet")
    comparacao = pd.read_parquet(RESULTADOS / "comparacao.parquet")
    assert len(dados["resultados"]) == len(tabela) == 42
    originais = tabela.set_index(["base", "modelo", "estrategia"])
    for registro in dados["resultados"]:
        chave = (registro["base"], registro["modelo"], registro["estrategia"])
        for metrica in (*METRICAS, "razao_previsto_observado"):
            assert registro[metrica] == originais.loc[chave, metrica]
        assert registro["contagens"] == contagens(comparacao, *chave)
        assert 0 < registro["prevalencia"] < 1
    assert dados["curvas"] == pd.read_parquet(
        RESULTADOS / "calibracao_curvas.parquet"
    ).to_dict("records")
    json.dumps(dados, allow_nan=False)


def test_projecoes_contem_todas_as_tecnicas_e_pontos_originais(dados):
    for base, projecao in dados["projecoes"].items():
        original = pd.read_parquet(RESULTADOS / f"amostra_{base}.parquet")
        assert projecao["total"] == len(original)
        assert set(projecao["tecnicas"]) == set(ESTRATEGIAS)
        assert len(projecao["tecnicas"]["nenhuma"]) == len(original)
        assert projecao["tecnicas"]["pesos"] == projecao["tecnicas"]["nenhuma"]
        assert projecao["tecnicas"]["limiar"] == projecao["tecnicas"]["nenhuma"]
        assert any(p[2] == "Sintético" for p in projecao["tecnicas"]["smote"])


@pytest.mark.parametrize("tipo", ["ausente", "corrompido", "vazio", "colunas", "nulo"])
def test_tabela_invalida_tem_erro_recuperavel(tmp_path, tipo):
    caminho = tmp_path / "tabela_final.parquet"
    if tipo == "corrompido":
        caminho.write_bytes(b"invalido")
    elif tipo != "ausente":
        tabela = {"base": []} if tipo == "vazio" else {
            "outra" if tipo == "colunas" else "base": [None if tipo == "nulo" else "brfss"]
        }
        pd.DataFrame(tabela).to_parquet(caminho)
    with pytest.raises(ErroResultados) as erro:
        ler_tabela(tmp_path, caminho.name, {"base"})
    assert erro.value.arquivo == caminho.name
    assert erro.value.comando == "python scripts/analisar.py"


def test_ausencia_de_amostras_e_curvas_preserva_metricas(tmp_path, dados):
    for nome in ("tabela_final.parquet", "comparacao.parquet"):
        shutil.copyfile(RESULTADOS / nome, tmp_path / nome)
    parcial = carregar_interface(tmp_path)
    assert parcial["resultados"] == dados["resultados"]
    assert parcial["curvas"] == []
    assert parcial["avisos"]
    assert all(p["erro"] and not p["tecnicas"] for p in parcial["projecoes"].values())


def test_amostra_pequena_nao_inventa_pontos_smote():
    amostra = pd.DataFrame({"x": range(12), "y": range(12), "alvo": [0] * 9 + [1] * 3})
    dados = projecoes_da_amostra(amostra)
    assert "smote" in dados["erros"]
    assert "smote_tomek" in dados["erros"]
    assert "smote" not in dados["tecnicas"]
    assert len(dados["tecnicas"]["nenhuma"]) == len(amostra)


def test_amostra_com_uma_classe_tem_aviso():
    assert "erro" in projecoes_da_amostra(pd.DataFrame({"x": [1, 2], "alvo": [0, 0]}))
