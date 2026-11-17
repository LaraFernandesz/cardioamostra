"""CardioAmostra: painel com as telas Laboratório, Comparativo e Metodologia.

    streamlit run app/painel.py

O painel só lê os arquivos prontos de resultados/ (gerados por
scripts/rodar_experimentos.py e scripts/analisar.py). Só a projeção PCA das
amostras de até 2.000 pessoas é calculada aqui, com cache. A interface em si é
um componente HTML/JS (app/interface/), que troca de tela pelo endereço (#comparativo).
"""

from pathlib import Path

import streamlit as st

PASTA = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CardioAmostra",
    page_icon=str(PASTA / "icone.svg"),
    layout="wide",
)
st.html("""
<style>
[data-testid="stMainBlockContainer"] {padding:0;max-width:none;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {gap:0;}
[data-testid="stHeader"] {display:none;}
[data-testid="stAppViewContainer"] {background:#faf9f6;}
[data-testid="stMainBlockContainer"] [data-testid="stAlert"],
[data-testid="stMainBlockContainer"] [data-testid="stCode"] {margin:20px;max-width:900px;}
</style>
""")

pagina = st.navigation(
    [
        st.Page(PASTA / "laboratorio.py", title="Laboratório", default=True),
        st.Page(PASTA / "comparativo.py", title="Comparativo"),
        st.Page(PASTA / "metodologia.py", title="Metodologia"),
    ],
    position="hidden",
)
pagina.run()
