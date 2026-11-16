"""Tela Laboratório: uma técnica por vez, antes e depois, comparada a nenhuma correção."""

from pathlib import Path

import comum
import plotly.graph_objects as go
import streamlit as st
from comum import COR, numero

from cardioamostra.apresentacao import (
    EXPLICACOES,
    MENOR_MELHOR,
    NOMES_BASES,
    NOMES_METRICAS,
    comparar_com_nenhuma,
    contagens,
    em_mil_pessoas,
    filtrar,
)
from cardioamostra.estrategias import obter

tabela = comum.resultados("tabela_final.parquet")
comparacao = comum.resultados("comparacao.parquet")

base = comum.seletor_base(tabela)
modelo = comum.seletor_modelo()
chave = comum.seletor_tecnica()
comum.rodape_lateral()

estrategia = obter(chave)
nome_base = NOMES_BASES[base]

# ------------------------------------------------------------- título ----
st.title(f"{estrategia.nome} na base {nome_base}")
if estrategia.reamostra:
    onde = " Acontece só nos dados de treino; o teste continua com pessoas reais."
else:
    onde = " Nenhum paciente é criado ou removido do treino."
st.html(f"<p class='ca-lead'>{estrategia.descricao}{onde}</p>")


# ------------------------------------------------------ antes / depois ----
def barra(titulo: str, total: int, partes: list[tuple[float, str]], maior: int) -> str:
    largura = 100 * total / maior
    blocos = "".join(
        f"<div style='width:{100 * n / total:.2f}%;background:{fundo}'></div>"
        for n, fundo in partes
        if n > 0
    )
    return (
        "<div style='display:flex;flex-direction:column;gap:6px'>"
        "<div style='display:flex;justify-content:space-between;font-size:12px'>"
        f"<b>{titulo}</b><span style='color:{COR['texto2']}'>{numero(total)} no treino</span>"
        "</div>"
        f"<div style='display:flex;height:16px;width:{largura:.1f}%;border-radius:3px;"
        f"overflow:hidden'>{blocos}</div></div>"
    )


c = contagens(comparacao, base, modelo, chave)
saudaveis_antes = c["linhas_antes"] - c["doentes_antes"]
saudaveis_depois = c["linhas_depois"] - c["doentes_depois"]
novos = max(c["doentes_depois"] - c["doentes_antes"], 0)
hachura = f"repeating-linear-gradient(135deg,{COR['doente']} 0 2px,{COR['doente_claro']} 2px 6px)"
maior = max(c["linhas_antes"], c["linhas_depois"])
antes = barra(
    "Antes",
    c["linhas_antes"],
    [(saudaveis_antes, COR["saudavel_barra"]), (c["doentes_antes"], COR["doente"])],
    maior,
)
depois = barra(
    f"Depois ({estrategia.nome})" if estrategia.reamostra else "Depois (igual)",
    c["linhas_depois"],
    [
        (saudaveis_depois, COR["saudavel_barra"]),
        (c["doentes_depois"] - novos, COR["doente"]),
        (novos, hachura),
    ],
    maior,
)
legenda_novos = "sintéticos" if chave in ("smote", "smote_tomek") else "repetidos"
legenda = (
    f"<span class='ca-rotulo'>■ <span style='color:{COR['saudavel_ponto']}'>sem doença</span>"
    f" · <span style='color:{COR['doente']}'>com doença</span>"
    + (f" · <span style='color:{COR['doente']}'>▨ {legenda_novos}</span>" if novos else "")
    + "</span>"
)
with st.container(border=True):
    st.html(
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:24px'>"
        f"{antes}{depois}</div>{legenda}"
    )
    limite = comum.metadados().get("limite_linhas")
    if base == "brfss" and limite:
        st.caption(
            f"Na BRFSS os experimentos usam um subconjunto estratificado de {numero(limite)} "
            "pessoas, como no artigo da IJACSA: o SMOTE + Tomek na base completa é lento demais. "
            "Os números são médias das 15 partes de treino."
        )

esquerda, direita = st.columns([1.15, 1], gap="medium")


# ------------------------------------------------------------ projeção ----
def grafico_projecao(pontos) -> go.Figure:
    fig = go.Figure()
    estilos = {
        "Sem doença": {"color": COR["saudavel_ponto"], "size": 5, "opacity": 0.55},
        "Descartado": {
            "color": COR["saudavel_barra"],
            "size": 5,
            "opacity": 0.6,
            "symbol": "circle-open",
        },
        "Sintético": {
            "color": COR["doente"],
            "size": 6,
            "opacity": 0.75,
            "symbol": "circle-open",
            "line": {"width": 1.2},
        },
        "Com doença (real)": {"color": COR["doente"], "size": 6, "opacity": 0.9},
    }
    for tipo, estilo in estilos.items():
        grupo = pontos[pontos["tipo"] == tipo]
        if grupo.empty:
            continue
        marcador = dict(estilo)
        if tipo == "Com doença (real)" and (grupo["copias"] > 1).any():
            marcador["size"] = (5 + 2.2 * (grupo["copias"] - 1) ** 0.5).tolist()
        fig.add_trace(
            go.Scattergl(
                x=grupo["pc1"],
                y=grupo["pc2"],
                mode="markers",
                name=tipo,
                marker=marcador,
                customdata=grupo["copias"],
                hovertemplate=(
                    f"{tipo}<br>cópias no treino: %{{customdata}}<extra></extra>"
                    if tipo == "Com doença (real)"
                    else f"{tipo}<extra></extra>"
                ),
            )
        )
    alcance = max(
        pontos["pc1"].quantile(0.99) - pontos["pc1"].quantile(0.01),
        pontos["pc2"].quantile(0.99) - pontos["pc2"].quantile(0.01),
    )
    passo = max(round(alcance / 6), 1)
    comum.papel_ecg(fig, 380, passo, passo)
    fig.update_xaxes(title="Componente 1", showticklabels=False)
    fig.update_yaxes(title="Componente 2", showticklabels=False)
    return fig


titulos = {
    "smote": "Onde os pacientes sintéticos caem",
    "smote_tomek": "Onde os sintéticos caem e quem o Tomek remove",
    "subamostragem": "Quem a subamostragem descarta",
    "sobreamostragem": "Doentes repetidos (maior = mais cópias)",
}
with esquerda, st.container(border=True):
    st.subheader(titulos.get(chave, "A amostra não muda"), anchor=False)
    if comum.minimo_smote(base, chave):
        pontos = comum.projecao(base, chave)
        if pontos is None:
            st.info(
                f"A amostra desta base (resultados/amostra_{base}.parquet) não está disponível. "
                "Gere com `python scripts/analisar.py`."
            )
        else:
            st.plotly_chart(
                grafico_projecao(pontos), width="stretch", config={"displayModeBar": False}
            )
            reais = int((pontos["tipo"] != "Sintético").sum())
            st.caption(
                f"Amostra de {numero(reais)} pessoas da base, projeção PCA. "
                "A técnica é aplicada só nesta amostra, para mostrar o que ela faz."
            )


# ----------------------------------------------------------- métricas ----
def selo(metrica: str, delta: float) -> str:
    casas = 3 if metrica == "brier" else 2
    if round(abs(delta), casas) == 0:
        texto, bom = "igual", True
    else:
        melhora = delta < 0 if metrica in MENOR_MELHOR else delta > 0
        sinal = "+" if delta > 0 else "−"
        prefixo = ("melhor " if melhora else "pior ") if metrica in MENOR_MELHOR else ""
        texto, bom = f"{prefixo}{sinal}{numero(abs(delta), casas)}", melhora
    cor, fundo = (COR["bom"], COR["bom_fundo"]) if bom else (COR["ruim"], COR["ruim_fundo"])
    return (
        f"<span style='font-size:12px;font-weight:700;color:{cor};background:{fundo};"
        f"padding:1px 6px;border-radius:4px;justify-self:end'>{texto}</span>"
    )


def linha_metrica(metrica: str, valor: float, delta: float | None) -> str:
    casas = 3 if metrica == "brier" else 2
    direita = selo(metrica, delta) if delta is not None else "<span></span>"
    return (
        f"<div style='padding:10px 0;border-bottom:1px solid {COR['borda']};display:grid;"
        "grid-template-columns:1fr auto;gap:2px 12px;align-items:baseline'>"
        f"<span style='font-weight:600'>{NOMES_METRICAS[metrica]}</span>"
        f"<span style='font-size:24px;font-weight:800;letter-spacing:-0.02em;"
        f"justify-self:end'>{numero(valor, casas)}</span>"
        f"<span style='font-size:12px;color:{COR['texto2']}'>{EXPLICACOES[metrica]}</span>"
        f"{direita}</div>"
    )


valores = comparar_com_nenhuma(tabela, base, modelo, chave)
referencia = chave == "nenhuma"
with direita, st.container(border=True):
    st.subheader("Resultado no teste", anchor=False)
    st.caption("Esta é a referência" if referencia else "Comparado a nenhuma correção")
    st.html(
        "".join(
            linha_metrica(m, valores[m]["valor"], None if referencia else valores[m]["delta"])
            for m in ("recall", "precisao", "roc_auc", "brier")
        )
    )
    linhas = filtrar(tabela, base, modelo)
    if chave not in ("nenhuma", "limiar"):
        limiar = linhas.loc["limiar"]
        st.html(
            f"<p style='margin:4px 0 0;font-size:13px;background:{COR['fundo']};"
            "border-radius:4px;padding:10px 12px'>O <b>ajuste de limiar</b> chega a recall de "
            f"{numero(limiar['recall'], 2)} com Brier de {numero(limiar['brier'], 3)}.</p>"
        )
    st.page_link(
        Path(__file__).resolve().parent / "comparativo.py", label="Comparar as 7 técnicas →"
    )


# --------------------------------------------------------- 1.000 pessoas ----
with st.expander("Em cada 1.000 pessoas do teste (aproximado)"):
    prevalencia = comum.tabela_prevalencia()[base]
    colunas = st.columns(2)
    for coluna, (titulo, est) in zip(
        colunas, [("Nenhuma correção", "nenhuma"), (estrategia.nome, chave)], strict=True
    ):
        m = em_mil_pessoas(prevalencia, linhas.loc[est, "recall"], linhas.loc[est, "precisao"])
        coluna.markdown(f"**{titulo}**")
        coluna.html(
            "<table style='width:100%;font-size:14px'>"
            f"<tr><td>Doentes encontrados</td><td style='text-align:right'><b>"
            f"{m['doentes_encontrados']}</b></td></tr>"
            f"<tr><td>Doentes perdidos</td><td style='text-align:right'><b>"
            f"{m['doentes_perdidos']}</b></td></tr>"
            f"<tr><td>Alarmes falsos</td><td style='text-align:right'><b>"
            f"{m['alarmes_falsos']}</b></td></tr>"
            f"<tr><td>Saudáveis liberados</td><td style='text-align:right'><b>"
            f"{m['saudaveis_corretos']}</b></td></tr></table>"
        )
    st.caption(
        "Contas feitas a partir da proporção de doentes da base e das médias de recall e "
        "precisão; por isso são aproximadas."
    )
