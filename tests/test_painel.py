"""Contrato do Streamlit; interações JavaScript são verificadas no navegador."""

import json
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
PAINEL = str(RAIZ / "app" / "painel.py")


@pytest.fixture(autouse=True)
def ambiente(monkeypatch):
    monkeypatch.delenv("CARDIOAMOSTRA_RESULTADOS", raising=False)
    for nome in ("comum", "interface"):
        sys.modules.pop(nome, None)
    yield
    for nome in ("comum", "interface"):
        sys.modules.pop(nome, None)


def abrir():
    return AppTest.from_file(PAINEL, default_timeout=120).run()


@pytest.mark.skipif(
    not (RAIZ / "resultados/tabela_final.parquet").exists(), reason="resultados não gerados"
)
def test_tres_paginas_recebem_resultados_reais_e_preservam_selecao():
    at = abrir()
    assert not at.exception
    assert not at.error
    componente = at.get("bidi_component")[0]
    dados = json.loads(componente.proto.json)
    assert dados["page"] == "laboratorio"
    assert len(dados["resultados"]) == 42
    assert len(dados["curvas"]) == 420
    assert componente.proto.isolate_styles
    selecao = {"base": "cleveland", "model": "gradient_boosting", "technique": "pesos"}
    at.session_state["cardio_selection"] = selecao
    for pagina in ("comparativo", "metodologia", "laboratorio"):
        at.switch_page(f"{pagina}.py").run()
        assert not at.exception
        dados = json.loads(at.get("bidi_component")[0].proto.json)
        assert dados["page"] == pagina
        assert dados["initial"] == selecao


def test_arquivo_ausente_diz_como_recuperar(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIOAMOSTRA_RESULTADOS", str(tmp_path))
    st.cache_data.clear()
    at = abrir()
    assert not at.exception
    assert "tabela_final.parquet" in at.error[0].value
    assert "scripts/analisar.py" in at.code[0].value
    assert at.button[0].label == "Tentar novamente"
    assert not at.get("bidi_component")
