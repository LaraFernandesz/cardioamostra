"""Peças compartilhadas pelas duas telas: leitura dos resultados, seletores e estilo."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from cardioamostra import experimento as exp  # noqa: E402
from cardioamostra.apresentacao import NOMES_BASES, projetar  # noqa: E402
from cardioamostra.estrategias import ESTRATEGIAS, MINIMO_DOENTES_SMOTE, obter  # noqa: E402

# CARDIOAMOSTRA_RESULTADOS troca a pasta (usado nos testes do estado "arquivo ausente").
RESULTADOS = Path(os.environ.get("CARDIOAMOSTRA_RESULTADOS") or RAIZ / "resultados")

COR = {
    "fundo": "#FFFBFA",
    "superficie": "#FFFFFF",
    "borda": "#E7E0E1",
    "texto": "#1B2236",
    "texto2": "#586178",
    "doente": "#B0124F",
    "doente_claro": "#F7DCE6",
    "saudavel_ponto": "#9AA3B5",
    "saudavel_barra": "#C9CFDA",
    "bom": "#1E6F5C",
    "bom_fundo": "#DDEFE9",
    "ruim": "#9A4A0B",
    "ruim_fundo": "#FBE9D9",
    "grade": "#FAEDEB",
    "grade_forte": "#F2D7D3",
}

# Comando que gera cada arquivo (mensagem do estado "falha ao ler os resultados").
COMANDOS = {
    "comparacao.parquet": "python scripts/rodar_experimentos.py --limite-linhas 50000",
    "tabela_final.parquet": "python scripts/analisar.py",
    "calibracao_curvas.parquet": "python scripts/analisar.py",
}

AVISO = (
    "Projeto educativo com dados públicos. Não serve para diagnóstico nem para avaliar o "
    "risco de um paciente."
)


# --------------------------------------------------------------------------- #
# Leitura dos resultados
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Carregando resultados…")
def _ler(nome: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTADOS / nome)


def resultados(nome: str) -> pd.DataFrame:
    """Lê um arquivo de resultados ou para a página explicando como gerá-lo."""
    try:
        return _ler(nome)
    except FileNotFoundError:
        st.error(
            f"Não encontrei **resultados/{nome}**. Gere o arquivo com o comando abaixo, "
            "na pasta do projeto, e recarregue a página."
        )
        st.code(COMANDOS.get(nome, "python scripts/analisar.py"), language="bash")
        st.stop()
        raise  # st.stop() interrompe; isto só satisfaz o verificador de tipos


@st.cache_data(show_spinner=False)
def metadados(nome: str = "comparacao.parquet") -> dict[str, Any]:
    caminho = RESULTADOS / nome
    return exp.ler_metadados(caminho) if caminho.exists() else {}


@st.cache_data(show_spinner=False)
def amostra(base: str) -> pd.DataFrame | None:
    caminho = RESULTADOS / f"amostra_{base}.parquet"
    return pd.read_parquet(caminho) if caminho.exists() else None


@st.cache_data(show_spinner="Calculando a projeção…")
def projecao(base: str, estrategia: str) -> pd.DataFrame | None:
    dados = amostra(base)
    if dados is None:
        return None
    X = dados.drop(columns="alvo")
    return projetar(X, dados["alvo"], obter(estrategia))


# --------------------------------------------------------------------------- #
# Seletores que guardam a escolha ao trocar de tela
# --------------------------------------------------------------------------- #
def _persistente(chave: str, padrao: Any) -> str:
    """Chave do widget que lembra a escolha mesmo quando a outra tela está aberta.

    O Streamlit apaga o estado de um widget que some da tela; por isso a escolha
    fica numa chave própria e é copiada para o widget antes de ele aparecer.
    """
    if chave not in st.session_state:
        st.session_state[chave] = padrao
    widget = f"_{chave}"
    st.session_state[widget] = st.session_state[chave]
    return widget


def _guardar(chave: str) -> None:
    st.session_state[chave] = st.session_state[f"_{chave}"]


def seletor_base(tabela: pd.DataFrame) -> str:
    bases = [b for b in NOMES_BASES if b in set(tabela["base"])]
    base = st.sidebar.selectbox(
        "Base de dados",
        bases,
        format_func=NOMES_BASES.get,
        key=_persistente("base", bases[0]),
        on_change=_guardar,
        args=("base",),
    )
    linhas = metadados().get("linhas_por_base", {}).get(base)
    prevalencia = tabela_prevalencia().get(base)
    if linhas and prevalencia is not None:
        st.sidebar.caption(
            f"{numero(linhas)} pessoas no experimento · "
            f"{numero(prevalencia * 100, 2)}% com doença"
        )
    return str(base)


def seletor_modelo() -> str:
    modelos = list(exp.MODELOS)
    return str(
        st.sidebar.selectbox(
            "Modelo",
            modelos,
            format_func=exp.MODELOS.get,
            key=_persistente("modelo", modelos[0]),
            on_change=_guardar,
            args=("modelo",),
        )
    )


def seletor_tecnica() -> str:
    chaves = list(ESTRATEGIAS)
    return str(
        st.sidebar.radio(
            "Técnica de amostragem",
            chaves,
            format_func=lambda c: ESTRATEGIAS[c].nome,
            key=_persistente("estrategia", "smote"),
            on_change=_guardar,
            args=("estrategia",),
        )
    )


def escolher(chave: str, valor: str) -> None:
    """Define uma escolha antes de trocar de tela (ex.: abrir técnica no Laboratório)."""
    st.session_state[chave] = valor


def rodape_lateral() -> None:
    st.sidebar.divider()
    st.sidebar.caption(AVISO)
    st.sidebar.caption(
        "Bases: BRFSS 2020 (Kaggle, CC0), Heart Failure Clinical Records e Heart Disease "
        "Cleveland (UCI, CC BY 4.0)."
    )


@st.cache_data(show_spinner=False)
def tabela_prevalencia() -> dict[str, float]:
    """Proporção de doentes de cada base, tirada das contagens do experimento."""
    comp = _ler("comparacao.parquet") if (RESULTADOS / "comparacao.parquet").exists() else None
    if comp is None:
        return {}
    filtro = comp["estrategia"].eq("nenhuma") & comp["metrica"].isin(
        ["linhas_treino", "positivos_treino"]
    )
    medias = comp[filtro].groupby(["base", "metrica"])["valor"].mean().unstack()
    return (medias["positivos_treino"] / medias["linhas_treino"]).to_dict()


def minimo_smote(base: str, estrategia: str) -> bool:
    """Estado "base pequena demais para SMOTE": avisa e devolve False se faltar doente."""
    dados = amostra(base)
    if dados is None or estrategia not in ("smote", "smote_tomek"):
        return True
    doentes = int(dados["alvo"].sum())
    if doentes >= MINIMO_DOENTES_SMOTE:
        return True
    st.warning(
        f"O SMOTE usa os {MINIMO_DOENTES_SMOTE - 1} vizinhos mais próximos de cada doente, "
        f"então precisa de pelo menos {MINIMO_DOENTES_SMOTE} doentes. Esta amostra tem "
        f"{doentes}."
    )
    return False


# --------------------------------------------------------------------------- #
# Formatação e gráficos
# --------------------------------------------------------------------------- #
def numero(valor: float, casas: int = 0) -> str:
    """Número no formato brasileiro: 241.373 e 0,76."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def papel_ecg(fig: go.Figure, altura: int, passo_x: float, passo_y: float) -> go.Figure:
    """Fundo de papel de eletrocardiograma: linha forte a cada passo, fina a cada 1/5."""
    eixo = {
        "showgrid": True,
        "gridcolor": COR["grade_forte"],
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": COR["borda"],
        "tickfont": {"color": COR["texto2"], "size": 11},
        "title": {"font": {"color": COR["texto2"], "size": 12}},
    }
    fig.update_layout(
        height=altura,
        margin={"l": 48, "r": 12, "t": 8, "b": 44},
        plot_bgcolor=COR["fundo"],
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Public Sans, sans-serif", "color": COR["texto"]},
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"size": 12}},
        hoverlabel={"font": {"family": "Public Sans, sans-serif"}},
    )
    fino = {"showgrid": True, "gridcolor": COR["grade"]}
    fig.update_xaxes(**eixo, dtick=passo_x, minor={**fino, "dtick": passo_x / 5})
    fig.update_yaxes(**eixo, dtick=passo_y, minor={**fino, "dtick": passo_y / 5})
    return fig


ESTILO = f"""
<style>
h1 {{ font-weight: 800 !important; letter-spacing: -0.02em; }}
h2, h3 {{ font-weight: 700 !important; }}
.stApp, .stMarkdown, td, th {{ font-variant-numeric: tabular-nums; }}
.ca-lead {{ color: {COR["texto2"]}; max-width: 72ch; margin-top: -0.5rem; }}
.ca-rotulo {{ font-size: 12px; color: {COR["texto2"]}; }}
</style>
"""
