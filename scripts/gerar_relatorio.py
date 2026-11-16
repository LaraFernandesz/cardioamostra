"""Gera relatorio/relatorio.pdf a partir de relatorio/relatorio.md.

    python scripts/gerar_relatorio.py

As tabelas do texto ({{tabela:...}}) são preenchidas direto de
resultados/tabela_final.parquet, para que nenhum número seja copiado à mão.
O PDF é impresso pelo Chrome via Playwright (usa o Chrome instalado; se não
houver, rode ``playwright install chromium``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

PASTA = RAIZ / "relatorio"
CORRIGEM = ["pesos", "subamostragem", "sobreamostragem", "smote", "smote_tomek"]
ORDEM = ["nenhuma", "limiar", *CORRIGEM]

CSS = """
@page { size: A4; margin: 16mm 16mm 18mm; }
body { font-family: 'Public Sans', 'Segoe UI', Arial, sans-serif; color: #1B2236;
       font-size: 9.5pt; line-height: 1.42; font-variant-numeric: tabular-nums; }
h1 { font-size: 18pt; line-height: 1.15; margin: 0 0 4pt; letter-spacing: -0.01em; }
h2 { font-size: 12.5pt; margin: 13pt 0 5pt; border-bottom: 1px solid #E7E0E1; padding-bottom: 3pt;
     break-after: avoid; page-break-after: avoid; }
p, li { margin: 0 0 5pt; }
ol li, ul li { margin-bottom: 4pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 8.8pt;
        page-break-inside: avoid; }
th { text-align: left; color: #586178; font-weight: 600; border-bottom: 1px solid #C9CFDA;
     padding: 3pt 5pt; }
td { border-bottom: 1px solid #E7E0E1; padding: 3pt 5pt; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
img { display: block; width: 86%; margin: 6pt auto 2pt; page-break-inside: avoid; }
code, pre { font-family: Consolas, monospace; font-size: 8.8pt; }
pre { background: #FFFBFA; border: 1px solid #E7E0E1; padding: 6pt 8pt; border-radius: 4px; }
strong { font-weight: 700; }
"""


def _n(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",").replace("-", "−")


def _faixa(serie, casas: int) -> str:
    menor, maior = _n(serie.min(), casas), _n(serie.max(), casas)
    return menor if menor == maior else f"{menor} a {maior}"


def tabela_base(tabela, base: str, modelo: str) -> str:
    from cardioamostra.estrategias import ESTRATEGIAS

    t = tabela[(tabela["base"] == base) & (tabela["modelo"] == modelo)].set_index("estrategia")
    linhas = [
        "| Técnica | Recall | Precisão | F1 | ROC-AUC | Brier | Prev. ÷ real | Intercepto |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for chave in ORDEM:
        r = t.loc[chave]
        nome = ESTRATEGIAS[chave].nome
        if chave in ("nenhuma", "limiar"):
            nome = f"**{nome}**"
        linhas.append(
            f"| {nome} | {_n(r['recall'], 2)} | {_n(r['precisao'], 2)} | {_n(r['f1'], 2)} "
            f"| {_n(r['roc_auc'], 3)} | {_n(r['brier'], 3)} "
            f"| {_n(r['razao_previsto_observado'], 2)} | {_n(r['intercepto'], 2)} |"
        )
    return "\n".join(linhas)


def tabela_pequenas(tabela) -> str:
    from cardioamostra.apresentacao import NOMES_BASES
    from cardioamostra.experimento import MODELOS

    linhas = [
        "| Base · modelo | Recall sem correção | Recall com limiar | Recall das 5 correções "
        "| Brier sem correção | Brier com limiar | Brier das 5 correções |",
        "|---|---|---|---|---|---|---|",
    ]
    for base in ("insuficiencia", "cleveland"):
        for modelo, nome_modelo in MODELOS.items():
            t = tabela[(tabela["base"] == base) & (tabela["modelo"] == modelo)].set_index(
                "estrategia"
            )
            c = t.loc[CORRIGEM]
            linhas.append(
                f"| {NOMES_BASES[base].split()[0]} · {nome_modelo.lower()} "
                f"| {_n(t.loc['nenhuma', 'recall'], 2)} | {_n(t.loc['limiar', 'recall'], 2)} "
                f"| {_faixa(c['recall'], 2)} | {_n(t.loc['nenhuma', 'brier'], 3)} "
                f"| {_n(t.loc['limiar', 'brier'], 3)} | {_faixa(c['brier'], 3)} |"
            )
    return "\n".join(linhas)


def montar_markdown() -> str:
    import pandas as pd

    tabela = pd.read_parquet(RAIZ / "resultados" / "tabela_final.parquet")
    texto = (PASTA / "relatorio.md").read_text(encoding="utf-8")

    def trocar(m: re.Match[str]) -> str:
        partes = m.group(1).split(":")
        if partes[0] == "pequenas":
            return tabela_pequenas(tabela)
        return tabela_base(tabela, partes[0], partes[1])

    return re.sub(r"\{\{tabela:([a-z_:]+)\}\}", trocar, texto)


def main() -> int:
    import markdown
    from playwright.sync_api import sync_playwright

    corpo = markdown.markdown(montar_markdown(), extensions=["tables", "fenced_code"])
    html = (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        "<title>CardioAmostra — Relatório</title>"
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Public+Sans:"
        "wght@400;600;700;800&display=swap'>"
        f"<style>{CSS}</style></head><body>{corpo}</body></html>"
    )
    pagina_html = PASTA / "relatorio.html"
    pagina_html.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        try:
            navegador = p.chromium.launch(channel="chrome")
        except Exception:
            navegador = p.chromium.launch()
        pagina = navegador.new_page()
        pagina.goto(pagina_html.as_uri(), wait_until="networkidle")
        pagina.pdf(
            path=str(PASTA / "relatorio.pdf"),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=(
                "<div style='font-size:8px;color:#586178;width:100%;text-align:center'>"
                "CardioAmostra · <span class='pageNumber'></span> de "
                "<span class='totalPages'></span></div>"
            ),
            margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
        )
        navegador.close()
    pagina_html.unlink()
    print("Relatório gravado em", PASTA / "relatorio.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
