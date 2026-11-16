"""Gera as figuras do relatório e do README em figuras/.

    python scripts/gerar_figuras.py

Lê resultados/tabela_final.parquet, calibracao_curvas.parquet e
investigacao_smote_gb.csv (gerados por scripts/refazer_tudo.py).

Cores: azul, laranja e verde-água, as 3 primeiras da paleta categórica
validada para daltonismo (passam em todos os pares). Verde e laranja, como nas
telas de referência, não passam para quem tem protanopia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from cardioamostra.apresentacao import NOMES_BASES  # noqa: E402

RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"

AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXTO, TEXTO2, BORDA, FUNDO = "#1B2236", "#586178", "#E7E0E1", "#FFFBFA"
GRADE, GRADE_FORTE = "#FAEDEB", "#F2D7D3"
CORRIGEM = ["pesos", "subamostragem", "sobreamostragem", "smote", "smote_tomek"]
MODELOS = {"regressao_logistica": "Regressão logística", "gradient_boosting": "Gradient boosting"}

plt.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": FUNDO,
        "axes.edgecolor": BORDA,
        "axes.labelcolor": TEXTO2,
        "xtick.color": TEXTO2,
        "ytick.color": TEXTO2,
        "text.color": TEXTO,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "font.size": 10,
        "axes.formatter.use_locale": False,
    }
)


def virgula(eixo) -> None:
    eixo.xaxis.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    eixo.yaxis.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))


def papel_ecg(eixo, passo: float) -> None:
    eixo.set_axisbelow(True)
    eixo.xaxis.set_major_locator(plt.MultipleLocator(passo))
    eixo.yaxis.set_major_locator(plt.MultipleLocator(passo))
    eixo.xaxis.set_minor_locator(plt.MultipleLocator(passo / 5))
    eixo.yaxis.set_minor_locator(plt.MultipleLocator(passo / 5))
    eixo.grid(which="major", color=GRADE_FORTE, linewidth=0.8)
    eixo.grid(which="minor", color=GRADE, linewidth=0.5)
    eixo.tick_params(which="minor", length=0)


def recall_brier(tabela: pd.DataFrame) -> None:
    """Cada técnica como um ponto: quanto encontra (recall) × quanto erra a probabilidade."""
    fig, eixos = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True)
    for i, modelo in enumerate(MODELOS):
        for j, base in enumerate(NOMES_BASES):
            ax = eixos[i, j]
            t = tabela[(tabela["base"] == base) & (tabela["modelo"] == modelo)].set_index(
                "estrategia"
            )
            outras = t.loc[CORRIGEM]
            ax.scatter(
                outras["recall"], outras["brier"], s=64, color=LARANJA, edgecolor="white",
                linewidth=1.5, zorder=3, label="Reamostragem e pesos de classe",
            )
            for chave, rotulo in (("nenhuma", "Nenhuma"), ("limiar", "Limiar")):
                ax.scatter(
                    t.loc[chave, "recall"], t.loc[chave, "brier"], s=64, color=AZUL,
                    edgecolor="white", linewidth=1.5, zorder=4,
                    marker="o" if chave == "nenhuma" else "D",
                    label="Nenhuma correção" if chave == "nenhuma" else "Ajuste de limiar",
                )
                # "Nenhuma" à esquerda do ponto, exceto colado ao eixo (recall baixo)
                esquerda = chave == "nenhuma" and t.loc[chave, "recall"] > 0.3
                ax.annotate(
                    rotulo, (t.loc[chave, "recall"], t.loc[chave, "brier"]),
                    xytext=(-7 if esquerda else 7, -13), textcoords="offset points",
                    ha="right" if esquerda else "left", fontsize=9, color=TEXTO,
                )
            if base == "brfss" and modelo == "gradient_boosting":
                smote = t.loc["smote"]
                ax.annotate(
                    "SMOTE e\nSMOTE + Tomek", (smote["recall"], smote["brier"]),
                    xytext=(10, 6), textcoords="offset points", fontsize=9, color=TEXTO,
                )
            ax.set_xlim(0, 1)
            topo = max(0.05, float(t["brier"].max()) * 1.25)
            ax.set_ylim(0, topo)
            papel_ecg(ax, 0.25)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_minor_locator(plt.NullLocator())
            virgula(ax)
            if i == 0:
                ax.set_title(NOMES_BASES[base])
            if j == 0:
                ax.set_ylabel(f"{MODELOS[modelo]}\nBrier (menor é melhor)")
            if i == 1:
                ax.set_xlabel("Recall (doentes encontrados)")
    alcas, nomes = eixos[0, 0].get_legend_handles_labels()
    unicos = dict(zip(nomes, alcas, strict=True))
    fig.legend(unicos.values(), unicos.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Nas bases desbalanceadas, corrigir sobe o recall e piora o Brier; o limiar não piora",
        x=0.01, ha="left", fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(FIGURAS / "recall_brier.png", dpi=160)
    plt.close(fig)


def calibracao(curvas: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.9))
    series = [
        ("nenhuma", "Nenhuma correção e ajuste de limiar", AZUL),
        ("smote", "SMOTE", LARANJA),
        ("subamostragem", "Subamostragem", AQUA),
    ]
    for ax, modelo in zip(eixos, MODELOS, strict=True):
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color=TEXTO2, linewidth=1, label="Calibração perfeita"
        )
        c = curvas[(curvas["base"] == "brfss") & (curvas["modelo"] == modelo)]
        maior = c["pessoas"].max()
        for chave, nome, cor in series:
            s = c[c["estrategia"] == chave]
            ax.plot(s["prevista"], s["observada"], color=cor, linewidth=2, label=nome, zorder=3)
            ax.scatter(
                s["prevista"], s["observada"], s=20 + 160 * (s["pessoas"] / maior) ** 0.5,
                color=cor, edgecolor="white", linewidth=1.5, zorder=4,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        papel_ecg(ax, 0.25)
        virgula(ax)
        ax.set_title(f"BRFSS 2020 · {MODELOS[modelo]}")
        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Proporção observada de doentes")
    alcas, nomes = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, nomes, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Curvas de calibração (10 faixas; ponto maior = mais pessoas na faixa)",
        x=0.01, ha="left", fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(FIGURAS / "calibracao_brfss.png", dpi=160)
    plt.close(fig)


def smote_gb(investigacao: pd.DataFrame) -> None:
    nomes = {
        "nenhuma": "Nenhuma correção",
        "sobreamostragem": "Sobreamostragem",
        "smote": "SMOTE",
        "smote_arredondado": "SMOTE com inteiros arredondados",
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    ordem = list(nomes)[::-1]
    altura = 0.36
    for k, (modelo, cor) in enumerate(
        (("regressao_logistica", AZUL), ("gradient_boosting", LARANJA))
    ):
        valores = investigacao.loc[modelo].reindex(ordem)["recall"]
        posicoes = [i + (k - 0.5) * (altura + 0.04) for i in range(len(ordem))]
        ax.barh(posicoes, valores, height=altura, color=cor, label=MODELOS[modelo])
        for y, v in zip(posicoes, valores, strict=True):
            ax.text(v + 0.01, y, f"{v:.2f}".replace(".", ","), va="center", fontsize=9)
    ax.set_yticks(range(len(ordem)), [nomes[o] for o in ordem])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall no teste (validação cruzada de 5 partes, BRFSS 50.000)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.grid(axis="x", color=GRADE_FORTE, linewidth=0.8)
    ax.set_axisbelow(True)
    alcas, rotulos = ax.get_legend_handles_labels()
    ax.legend(
        alcas[::-1], rotulos[::-1], loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Arredondar os sintéticos devolve o recall do gradient boosting",
        x=0.01, ha="left", fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURAS / "smote_gradient_boosting.png", dpi=160)
    plt.close(fig)


def main() -> int:
    FIGURAS.mkdir(exist_ok=True)
    tabela = pd.read_parquet(RESULTADOS / "tabela_final.parquet")
    curvas = pd.read_parquet(RESULTADOS / "calibracao_curvas.parquet")
    investigacao = pd.read_csv(
        RESULTADOS / "investigacao_smote_gb.csv", sep=";", decimal=",", index_col=[0, 1]
    )
    recall_brier(tabela)
    calibracao(curvas)
    smote_gb(investigacao)
    print("Figuras gravadas em", FIGURAS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
