"""Leitura com cache e erros recuperaveis para o painel Streamlit."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from cardioamostra.interface import ErroResultados, carregar_interface  # noqa: E402

RESULTADOS = Path(os.environ.get("CARDIOAMOSTRA_RESULTADOS") or RAIZ / "resultados")


@st.cache_data(show_spinner="Preparando resultados e projeções da amostra…")
def _dados(pasta: str, assinatura: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    """A assinatura invalida o cache quando os arquivos de origem mudam."""
    return carregar_interface(Path(pasta))


def dados_interface() -> dict[str, Any]:
    arquivos = sorted(RESULTADOS.glob("*.parquet"))
    assinatura = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in arquivos)
    try:
        return _dados(str(RESULTADOS.resolve()), assinatura)
    except ErroResultados as erro:
        st.error(str(erro))
        st.caption("Reconstrua os resultados e tente novamente. Nenhum dado foi alterado.")
        st.code(erro.comando, language="bash")
        if st.button("Tentar novamente"):
            _dados.clear()
            st.rerun()
        st.stop()
        raise
