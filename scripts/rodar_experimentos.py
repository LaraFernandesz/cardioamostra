"""Roda os experimentos e grava os resultados em resultados/.

Exemplos:
    python scripts/rodar_experimentos.py --baseline
    python scripts/rodar_experimentos.py
    python scripts/rodar_experimentos.py --bases brfss --limite-linhas 50000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

BASES_PADRAO = ["brfss", "insuficiencia", "cleveland"]


def versao_do_codigo() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        sujo = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{commit}-modificado" if sujo else commit
    except (OSError, subprocess.CalledProcessError):
        return "sem-git"


def ler_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    from cardioamostra.estrategias import ESTRATEGIAS
    from cardioamostra.experimento import MODELOS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true", help="só 'nenhuma correção'")
    parser.add_argument("--bases", nargs="+", default=BASES_PADRAO)
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    parser.add_argument(
        "--estrategias", nargs="+", default=list(ESTRATEGIAS), choices=list(ESTRATEGIAS)
    )
    parser.add_argument(
        "--limite-linhas",
        type=int,
        default=None,
        help="subconjunto estratificado para bases maiores que isso (ex.: 50000 na BRFSS)",
    )
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados")
    return parser.parse_args(argv)


def ordenar(chaves: list[str]) -> list[str]:
    """Técnicas de reamostragem antes do limiar, que depende do recall delas."""
    from cardioamostra.estrategias import obter

    return sorted(chaves, key=lambda chave: obter(chave).ajusta_limiar)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from cardioamostra import experimento as exp
    from cardioamostra.dados import carregar
    from cardioamostra.estrategias import MINIMO_DOENTES_SMOTE, de_reamostragem, obter

    args = ler_argumentos(argv)
    chaves = ["nenhuma"] if args.baseline else ordenar(args.estrategias)
    if "limiar" in chaves:
        exigidas = {e.chave for e in de_reamostragem()}
        faltando = exigidas - set(chaves)
        if faltando:
            print(
                "O ajuste de limiar precisa das técnicas de reamostragem na mesma execução. "
                f"Faltam: {', '.join(sorted(faltando))}",
                file=sys.stderr,
            )
            return 2

    resultados = []
    predicoes = []
    linhas_por_base: dict[str, int] = {}
    recalls_alvo: dict[str, float] = {}

    for base in args.bases:
        X, y = carregar(base)
        if args.limite_linhas:
            X, y = exp.subamostrar_estratificado(X, y, args.limite_linhas)
        linhas_por_base[base] = int(len(X))
        doentes = int(y.sum())
        print(f"\n{base}: {len(X)} linhas, {doentes} doentes ({doentes / len(X):.2%})")
        if doentes < MINIMO_DOENTES_SMOTE * exp.PARTES:
            print(f"  aviso: poucos doentes para SMOTE com {exp.PARTES} partes", file=sys.stderr)

        for modelo in args.modelos:
            da_configuracao: list[pd.DataFrame] = []
            for chave in chaves:
                estrategia = obter(chave)
                recall_alvo = None
                if estrategia.ajusta_limiar:
                    recall_alvo = exp.recall_alvo_para_limiar(
                        exp.concatenar(da_configuracao), base, modelo
                    )
                    recalls_alvo[f"{base}/{modelo}"] = round(recall_alvo, 4)

                inicio = time.perf_counter()
                tabela, probabilidades = exp.avaliar(
                    X, y, estrategia, modelo, base, recall_alvo=recall_alvo
                )
                segundos = time.perf_counter() - inicio
                da_configuracao.append(tabela)
                predicoes.append(probabilidades)

                medias = tabela.groupby("metrica")["valor"].mean()
                print(
                    f"  {modelo:<20} {chave:<16} recall {medias['recall']:.3f}  "
                    f"precisão {medias['precisao']:.3f}  ROC-AUC {medias['roc_auc']:.3f}  "
                    f"Brier {medias['brier']:.3f}  ({segundos:.0f} s)"
                )
            resultados.extend(da_configuracao)

    tabela_final = exp.concatenar(resultados)
    metadados = exp.metadados_base(
        versao_do_codigo(),
        {
            "bases": args.bases,
            "modelos": args.modelos,
            "estrategias": chaves,
            "limite_linhas": args.limite_linhas,
            "linhas_por_base": linhas_por_base,
            "recall_alvo_limiar": recalls_alvo,
        },
    )

    nome = "baseline" if args.baseline else "comparacao"
    exp.salvar_parquet(tabela_final, args.saida / f"{nome}.parquet", metadados)
    exp.salvar_parquet(
        exp.concatenar(predicoes) if predicoes else tabela_final.iloc[0:0],
        args.saida / f"predicoes_{nome}.parquet",
        metadados,
    )

    configuracoes = tabela_final[["base", "modelo", "estrategia"]].drop_duplicates()
    print(f"\n{len(configuracoes)} configurações gravadas em {args.saida / f'{nome}.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
