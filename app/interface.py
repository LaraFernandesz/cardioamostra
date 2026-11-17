"""Componente Streamlit da interface aprovada; sem servidor frontend separado."""

from pathlib import Path

import comum
import streamlit as st
import streamlit.components.v2 as components

PASTA = Path(__file__).resolve().parent / "interface"


@st.cache_resource
def componente(assinatura: tuple[tuple[str, int], ...]):
    return components.component(
        "cardioamostra_interface",
        html=(PASTA / "index.html").read_text(encoding="utf-8"),
        css=(PASTA / "style.css").read_text(encoding="utf-8")
        + (PASTA / "compact.css").read_text(encoding="utf-8")
        + (PASTA / "integracao.css").read_text(encoding="utf-8"),
        js=(PASTA / "app.js").read_text(encoding="utf-8"),
        isolate_styles=True,
    )


def renderizar(pagina: str) -> None:
    dados = comum.dados_interface()
    selecao = st.session_state.get("cardio_selection", {})

    def guardar_selecao() -> None:
        # Uma chave independente sobrevive à troca de página nativa do Streamlit.
        valor = st.session_state.get("cardio_ui", {}).get("selection")
        if valor:
            st.session_state["cardio_selection"] = valor

    assinatura = tuple((p.name, p.stat().st_mtime_ns) for p in sorted(PASTA.iterdir()))
    componente(assinatura)(
        data={**dados, "initial": selecao, "page": pagina},
        key="cardio_ui",
        on_selection_change=guardar_selecao,
        height="content",
    )
