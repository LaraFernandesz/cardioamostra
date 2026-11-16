"""Tela Comparativo: ranking das 7 técnicas e curvas de calibração sobrepostas."""

from pathlib import Path

import comum
import plotly.graph_objects as go
import streamlit as st
from comum import COR, numero

from cardioamostra.apresentacao import NOMES_METRICAS, ranking, resumo_comparativo
from cardioamostra.estrategias import ESTRATEGIAS

tabela = comum.resultados("tabela_final.parquet")
curvas = comum.resultados("calibracao_curvas.parquet")

base = comum.seletor_base(tabela)
modelo = comum.seletor_modelo()
ordem = ["brier", "recall", "precisao", "f1", "pr_auc", "roc_auc"]
metrica = st.sidebar.selectbox(
    "Ordenar por",
    ordem,
    format_func=lambda m: NOMES_METRICAS[m] + (" (menor é melhor)" if m == "brier" else ""),
)
comum.rodape_lateral()

st.title("Mesmo recall, probabilidades diferentes")
st.html(f"<p class='ca-lead'>{resumo_comparativo(tabela, base, modelo)}</p>")

esquerda, direita = st.columns([1.5, 1], gap="medium")

# ------------------------------------------------------------- ranking ----
r = ranking(tabela, base, modelo, metrica)
brier_nenhuma = float(r.loc["nenhuma", "brier"])


def celula_barra(valor: float, casas: int, largura: float, cor: str) -> str:
    return (
        "<td><div style='display:flex;align-items:center;gap:8px'>"
        f"<div style='width:{largura:.0f}px;height:8px;background:{cor};border-radius:2px'></div>"
        f"{numero(valor, casas)}</div></td>"
    )


linhas_html = []
maior_brier = float(r["brier"].max())
for chave, linha in r.iterrows():
    destaque = f"background:{COR['bom_fundo']}55;" if chave == "limiar" else ""
    peso = 700 if chave == "limiar" else 600
    cor_brier = COR["bom"] if linha["brier"] <= brier_nenhuma * 1.05 else COR["ruim"]
    razao = linha["razao_previsto_observado"]
    linhas_html.append(
        f"<tr style='{destaque}'><td style='font-weight:{peso}'>{linha['tecnica']}</td>"
        + celula_barra(linha["recall"], 2, 32 * linha["recall"], COR["doente"])
        + f"<td>{numero(linha['precisao'], 2)}</td><td>{numero(linha['roc_auc'], 2)}</td>"
        + celula_barra(linha["brier"], 3, 32 * linha["brier"] / maior_brier, cor_brier)
        + f"<td>{numero(razao, 2)}×</td></tr>"
    )

estilo_tabela = (
    "<style>.ca-rank{width:100%;border-collapse:collapse}"
    f".ca-rank th{{white-space:nowrap;text-align:left;font-size:12px;color:{COR['texto2']};"
    f"font-weight:600;padding:8px 6px;border-bottom:1px solid {COR['borda']}}}"
    f".ca-rank td{{white-space:nowrap;padding:10px 6px;font-size:13px;"
    f"border-bottom:1px solid {COR['borda']}}}"
    ".ca-rolagem{overflow-x:auto}</style>"
)
with esquerda, st.container(border=True):
    st.subheader("Ranking", anchor=False)
    st.html(
        estilo_tabela
        + "<div class='ca-rolagem'><table class='ca-rank'><thead><tr><th>Técnica</th>"
        "<th>Recall</th><th>Precisão</th><th>ROC-AUC</th><th>Brier</th>"
        "<th title='Risco médio previsto ÷ proporção real de doentes'>Prev. ÷ real</th>"
        "</tr></thead><tbody>" + "".join(linhas_html) + "</tbody></table></div>"
    )
    st.caption(
        "Médias das 15 rodadas da validação cruzada 5 × 3. *Prev. ÷ real*: risco médio que "
        "o modelo diz ÷ proporção real de doentes; acima de 1, ele exagera o risco."
    )
    abrir, botao = st.columns([3, 1], vertical_alignment="bottom")
    escolhida = abrir.selectbox(
        "Abrir uma técnica no Laboratório",
        list(r.index),
        format_func=lambda c: ESTRATEGIAS[c].nome,
    )
    if botao.button("Abrir", width="stretch"):
        comum.escolher("estrategia", escolhida)
        st.switch_page(Path(__file__).parent / "laboratorio.py")


# ---------------------------------------------------------- calibração ----
CORES_CURVAS = {
    "nenhuma": COR["bom"],
    "pesos": "#586178",
    "subamostragem": "#0E7490",
    "sobreamostragem": "#6D28D9",
    "smote": COR["ruim"],
    "smote_tomek": "#C2410C",
}
NOMES_CURVAS = {
    "nenhuma": "Nenhuma e limiar",
    **{k: ESTRATEGIAS[k].nome for k in CORES_CURVAS if k != "nenhuma"},
}

with direita, st.container(border=True):
    st.subheader("Calibração", anchor=False)
    mostrar = st.multiselect(
        "Técnicas no gráfico",
        list(CORES_CURVAS),
        default=["nenhuma", "smote", "subamostragem"],
        format_func=NOMES_CURVAS.get,
        label_visibility="collapsed",
    )
    filtro = (curvas["base"] == base) & (curvas["modelo"] == modelo)
    mais_gente = float(curvas.loc[filtro, "pessoas"].max())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": COR["texto2"], "dash": "dash", "width": 1.2},
            name="Calibração perfeita",
            hoverinfo="skip",
        )
    )
    for chave in mostrar:
        c = curvas[filtro & (curvas["estrategia"] == chave)]
        fig.add_trace(
            go.Scatter(
                x=c["prevista"],
                y=c["observada"],
                mode="lines+markers",
                name=NOMES_CURVAS[chave],
                line={"color": CORES_CURVAS[chave], "width": 2.4},
                # ponto maior = mais pessoas na faixa (faixas pequenas oscilam mais)
                marker={"size": (5 + 11 * (c["pessoas"] / mais_gente) ** 0.5).tolist()},
                customdata=c[["pessoas"]],
                hovertemplate=(
                    "previsto %{x:.2f} · observado %{y:.2f}<br>%{customdata[0]} pessoas"
                    "<extra>" + NOMES_CURVAS[chave] + "</extra>"
                ),
            )
        )
    comum.papel_ecg(fig, 400, 0.25, 0.25)
    fig.update_xaxes(title="Probabilidade prevista", range=[0, 1])
    fig.update_yaxes(title="Observado", range=[0, 1], scaleanchor="x", scaleratio=1)
    fig.update_layout(separators=",.", legend={"y": -0.22})
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Quanto mais perto da diagonal, mais a probabilidade dita pelo modelo bate com a "
        "realidade. Faixas de 0,1 de probabilidade; ponto maior = mais pessoas na faixa "
        "(faixas com pouca gente oscilam mais). Cada pessoa aparece uma vez "
        "(1ª repetição da validação cruzada). O ajuste de limiar não muda as probabilidades, "
        "por isso a curva dele é a mesma da nenhuma correção."
    )
