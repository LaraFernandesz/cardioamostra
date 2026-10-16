"""Baixa as 3 bases para dados/brutos/.

    python scripts/baixar_dados.py

* Cleveland e Insuficiência cardíaca: pacote ``ucimlrepo`` (sem login).
* BRFSS 2020: API do Kaggle. Precisa do token em ~/.kaggle/kaggle.json
  (Windows: C:\\Users\\<você>\\.kaggle\\kaggle.json). Gere em
  kaggle.com > Settings > API > Create New Token.

Se algo falhar, baixe manualmente pelos links de dados/brutos/LEIA-ME.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parents[1] / "dados" / "brutos"


def baixar_uci() -> None:
    from ucimlrepo import fetch_ucirepo

    cle = fetch_ucirepo(id=45)
    df = cle.data.features.copy()
    df["num"] = cle.data.targets.iloc[:, 0]
    df.to_csv(PASTA / "processed.cleveland.data", header=False, index=False, na_rep="?")
    print(f"Cleveland: {len(df)} linhas")

    hf = fetch_ucirepo(id=519)
    df = hf.data.features.copy()
    df["DEATH_EVENT"] = hf.data.targets.iloc[:, 0]
    df.to_csv(PASTA / "heart_failure_clinical_records_dataset.csv", index=False)
    print(f"Insuficiência cardíaca: {len(df)} linhas")


def baixar_kaggle() -> None:
    destino = PASTA / "brfss"
    destino.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "kamilpytlak/personal-key-indicators-of-heart-disease",
            "-p",
            str(destino),
            "--unzip",
        ],
        check=True,
    )
    print("BRFSS baixada em", destino)


if __name__ == "__main__":
    PASTA.mkdir(parents=True, exist_ok=True)
    for nome, funcao in (("UCI", baixar_uci), ("Kaggle", baixar_kaggle)):
        try:
            funcao()
        except Exception as erro:  # noqa: BLE001
            print(f"[{nome}] falhou: {erro}. Baixe manualmente (veja dados/brutos/LEIA-ME.md).")
