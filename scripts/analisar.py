"""Gera a análise de calibração e os arquivos que o painel lê.

    python scripts/analisar.py

Lê ``resultados/comparacao.parquet`` e ``resultados/predicoes_comparacao.parquet``
(gerados por ``scripts/rodar_experimentos.py``) e grava em ``resultados/``:

* ``calibracao_curvas.parquet``: curva de 10 faixas de cada configuração;
* ``calibracao_resumo.parquet``: razão previsto/observado, intercepto e inclinação;
* ``tabela_final.parquet`` e ``tabela_final.csv``: médias das 15 rodadas + calibração;
* ``amostra_<base>.parquet``: até 2.000 pessoas de cada base (estratificado), para a
  projeção PCA do painel, que roda sem os dados brutos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

TAMANHO_AMOSTRA = 2000


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    from rodar_experimentos import versao_do_codigo

    from cardioamostra import calibracao as cal
    from cardioamostra import experimento as exp
    from cardioamostra.dados import BASES, carregar, disponivel

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resultados", type=Path, default=RAIZ / "resultados")
    parser.add_argument("--sem-amostras", action="store_true", help="não regrava as amostras")
    args = parser.parse_args(argv)
    pasta: Path = args.resultados

    faltando = [
        nome
        for nome in ("comparacao.parquet", "predicoes_comparacao.parquet")
        if not (pasta / nome).exists()
    ]
    if faltando:
        print(
            f"Faltam {', '.join(faltando)} em {pasta}. Gere com:\n"
            "    python scripts/rodar_experimentos.py --limite-linhas 50000",
            file=sys.stderr,
        )
        return 2

    comparacao = pd.read_parquet(pasta / "comparacao.parquet")
    predicoes = pd.read_parquet(pasta / "predicoes_comparacao.parquet")
    metadados = exp.ler_metadados(pasta / "comparacao.parquet")
    metadados["analise_versao_codigo"] = versao_do_codigo()

    curvas = cal.curvas(predicoes)
    resumo = cal.resumo(predicoes)
    final = cal.tabela_final(comparacao, resumo)
    exp.salvar_parquet(curvas, pasta / "calibracao_curvas.parquet", metadados)
    exp.salvar_parquet(resumo, pasta / "calibracao_resumo.parquet", metadados)
    exp.salvar_parquet(final, pasta / "tabela_final.parquet", metadados)
    final.round(4).to_csv(pasta / "tabela_final.csv", index=False, decimal=",", sep=";")
    print(f"Tabela final: {len(final)} configurações; curvas: {len(curvas)} faixas")

    if not args.sem_amostras:
        for base in BASES:
            if not disponivel(base):
                print(f"  {base}: dados brutos ausentes, amostra não gerada", file=sys.stderr)
                continue
            X, y = carregar(base)
            X, y = exp.subamostrar_estratificado(X, y, TAMANHO_AMOSTRA)
            amostra = X.assign(alvo=y.to_numpy())
            exp.salvar_parquet(amostra, pasta / f"amostra_{base}.parquet", metadados)
            print(f"  amostra_{base}.parquet: {len(amostra)} pessoas, {int(y.sum())} doentes")

    pd.set_option("display.width", 160)
    colunas = ["recall", "precisao", "roc_auc", "brier", "razao_previsto_observado", "intercepto"]
    print(final.set_index(cal.CHAVES)[colunas].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
