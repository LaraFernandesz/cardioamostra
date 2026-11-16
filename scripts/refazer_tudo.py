"""Refaz todos os resultados do projeto, na ordem certa.

    python scripts/refazer_tudo.py

Precisa das 3 bases em dados/brutos/ (python scripts/baixar_dados.py). Leva
cerca de 20 minutos num computador comum; quase todo o tempo é o SMOTE + Tomek
na BRFSS. Ao final, resultados/ tem os mesmos arquivos que estão no repositório.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

PASSOS = [
    ("Baseline (Fase 2)", ["rodar_experimentos.py", "--baseline"]),
    ("As 7 estratégias (Fase 3)", ["rodar_experimentos.py", "--limite-linhas", "50000"]),
    ("Calibração, tabela final e amostras (Fase 4)", ["analisar.py"]),
    ("Investigação SMOTE × gradient boosting", ["investigar_smote_gb.py"]),
]


def main() -> int:
    for titulo, (script, *argumentos) in PASSOS:
        print(f"\n=== {titulo} ===", flush=True)
        inicio = time.perf_counter()
        comando = [sys.executable, str(RAIZ / "scripts" / script), *argumentos]
        if subprocess.run(comando, cwd=RAIZ).returncode != 0:
            print(f"Falhou: {' '.join(comando)}", file=sys.stderr)
            return 1
        print(f"({time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
