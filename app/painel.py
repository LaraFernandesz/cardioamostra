"""CardioAmostra: painel com as telas Laboratório e Comparativo.

    streamlit run app/painel.py

O painel só lê os arquivos prontos de resultados/ (gerados por
scripts/rodar_experimentos.py e scripts/analisar.py). Só a projeção PCA da
amostra de 2.000 pessoas é calculada aqui, com cache.
"""

from pathlib import Path

import comum
import streamlit as st

PASTA = Path(__file__).parent

st.set_page_config(
    page_title="CardioAmostra",
    page_icon=str(PASTA / "icone.svg"),
    layout="wide",
)
st.logo(str(PASTA / "logo.svg"), size="large", icon_image=str(PASTA / "icone.svg"))
st.html(comum.ESTILO)

pagina = st.navigation(
    [
        st.Page(PASTA / "laboratorio.py", title="Laboratório", default=True),
        st.Page(PASTA / "comparativo.py", title="Comparativo"),
    ]
)
pagina.run()
