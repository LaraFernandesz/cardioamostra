"""Testes do painel: as duas telas rodam sem erro, sem abrir o navegador.

Usam os arquivos de resultados/ que estão no repositório (pulados se faltarem).
"""

import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
PAINEL = str(RAIZ / "app" / "painel.py")
NECESSARIOS = ["tabela_final.parquet", "comparacao.parquet", "calibracao_curvas.parquet"]

pytestmark = pytest.mark.skipif(
    not all((RAIZ / "resultados" / n).exists() for n in NECESSARIOS),
    reason="resultados/ ainda não foi gerado (python scripts/refazer_tudo.py)",
)


@pytest.fixture(autouse=True)
def painel_limpo(monkeypatch):
    """Cada teste começa sem cache e relê comum.py (que guarda a pasta de resultados)."""
    monkeypatch.delenv("CARDIOAMOSTRA_RESULTADOS", raising=False)
    st.cache_data.clear()
    sys.modules.pop("comum", None)
    yield
    st.cache_data.clear()
    sys.modules.pop("comum", None)


def abrir() -> AppTest:
    return AppTest.from_file(PAINEL, default_timeout=120).run()


def test_laboratorio_abre_com_smote_na_brfss():
    at = abrir()
    assert not at.exception
    assert at.title[0].value == "SMOTE na base BRFSS 2020"
    assert not at.error


@pytest.mark.parametrize("base", ["brfss", "insuficiencia", "cleveland"])
def test_laboratorio_todas_as_tecnicas(base):
    at = abrir()
    at.sidebar.selectbox[0].set_value(base).run()
    for tecnica in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(tecnica).run()
        assert not at.exception, (base, tecnica)
        assert not at.error, (base, tecnica)


def test_comparativo_abre_e_ordena():
    at = abrir()
    at.switch_page("comparativo.py").run()
    assert not at.exception
    assert at.title[0].value == "Mesmo recall, probabilidades diferentes"
    for metrica in at.sidebar.selectbox[2].options:
        at.sidebar.selectbox[2].set_value(metrica).run()
        assert not at.exception
    comparar = next(s for s in at.selectbox if s.label.startswith("Comparar"))
    for tecnica in comparar.options:
        comparar.set_value(tecnica).run()
        assert not at.exception


def test_escolha_da_base_vale_nas_duas_telas():
    at = abrir()
    at.sidebar.selectbox[0].set_value("cleveland").run()
    at.switch_page("comparativo.py").run()
    assert at.sidebar.selectbox[0].value == "cleveland"


def test_arquivo_ausente_diz_qual_comando_rodar(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIOAMOSTRA_RESULTADOS", str(tmp_path))
    at = abrir()
    assert not at.exception
    assert "tabela_final.parquet" in at.error[0].value
    assert "scripts/analisar.py" in at.code[0].value
