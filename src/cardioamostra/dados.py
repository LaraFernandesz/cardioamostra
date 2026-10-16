"""Carga, limpeza e codificação das 3 bases do CardioAmostra.

Bases (arquivos brutos em ``dados/brutos/``, ou na pasta indicada pela variável
de ambiente ``CARDIOAMOSTRA_DADOS``):

* ``brfss``      -> heart_2020_cleaned.csv (Kaggle, K. Pytlak; BRFSS 2020)
* ``insuficiencia`` -> heart_failure_clinical_records_dataset.csv (UCI id 519)
* ``cleveland``  -> processed.cleveland.data (UCI id 45)

Os arquivos podem estar em subpastas: a busca é recursiva pelo nome.

Uso típico::

    from cardioamostra.dados import carregar
    X, y = carregar("brfss")

Decisões documentadas
---------------------
* BRFSS: duplicatas removidas ANTES de codificar (esperado 301.717 linhas,
  9,04% positivos). ``AgeCategory`` e ``GenHealth`` viram ordinais; as demais
  categóricas Yes/No viram 0/1 e as multiclasse recebem one-hot.
* Insuficiência cardíaca: alvo ``DEATH_EVENT``. A coluna ``time`` (tempo de
  acompanhamento) é removida por padrão, pois só é conhecida depois do
  desfecho e causaria vazamento (quem morre cedo tem acompanhamento curto).
* Cleveland: alvo ``num > 0``. Os 6 valores ausentes (4 em ``ca``, 2 em
  ``thal``) são preenchidos com a moda, para manter as 303 linhas; ``cp``,
  ``restecg``, ``slope`` e ``thal`` recebem one-hot.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd

NomeBase = Literal["brfss", "insuficiencia", "cleveland"]
BASES: tuple[NomeBase, ...] = ("brfss", "insuficiencia", "cleveland")

ARQUIVOS: dict[str, str] = {
    "brfss": "heart_2020_cleaned.csv",
    "insuficiencia": "heart_failure_clinical_records_dataset.csv",
    "cleveland": "processed.cleveland.data",
}

# Valores esperados, usados nos testes e no README.
ESPERADO: dict[str, dict[str, float]] = {
    "brfss": {"linhas_brutas": 319_795, "linhas": 301_717, "prevalencia": 0.0904},
    "insuficiencia": {"linhas_brutas": 299, "linhas": 299, "prevalencia": 96 / 299},
    "cleveland": {"linhas_brutas": 303, "linhas": 303, "prevalencia": 139 / 303},
}

ORDEM_IDADE = [
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80 or older",
]
ORDEM_SAUDE = ["Poor", "Fair", "Good", "Very good", "Excellent"]

COLUNAS_CLEVELAND = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


# --------------------------------------------------------------------------- #
# Localização dos arquivos
# --------------------------------------------------------------------------- #
def pasta_dados() -> Path:
    """Pasta dos arquivos brutos (variável CARDIOAMOSTRA_DADOS ou dados/brutos)."""
    env = os.environ.get("CARDIOAMOSTRA_DADOS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "dados" / "brutos"


def caminho(base: NomeBase, pasta: Path | None = None) -> Path:
    """Acha o arquivo bruto da base (busca recursiva pelo nome)."""
    pasta = pasta or pasta_dados()
    nome = ARQUIVOS[base]
    achados = sorted(pasta.rglob(nome)) if pasta.exists() else []
    if not achados:
        raise FileNotFoundError(
            f"Arquivo '{nome}' da base '{base}' não encontrado em {pasta}. "
            "Veja dados/brutos/LEIA-ME.md para saber onde baixar."
        )
    return achados[0]


def disponivel(base: NomeBase, pasta: Path | None = None) -> bool:
    try:
        caminho(base, pasta)
        return True
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# Leitura bruta (sem codificar)
# --------------------------------------------------------------------------- #
def ler_bruto(base: NomeBase, pasta: Path | None = None) -> pd.DataFrame:
    """Lê o arquivo original, sem limpeza."""
    arq = caminho(base, pasta)
    if base == "cleveland":
        return pd.read_csv(arq, header=None, names=COLUNAS_CLEVELAND, na_values="?")
    return pd.read_csv(arq)


# --------------------------------------------------------------------------- #
# Limpeza (alvo 0/1, duplicatas, ausentes) — ainda legível, sem one-hot
# --------------------------------------------------------------------------- #
def limpar_brfss(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={"HeartDisease": "alvo"})
    df["alvo"] = _sim_nao(df["alvo"], "HeartDisease")
    return df


def limpar_insuficiencia(df: pd.DataFrame, manter_tempo: bool = False) -> pd.DataFrame:
    df = df.rename(columns={"DEATH_EVENT": "alvo"}).copy()
    if not manter_tempo and "time" in df.columns:
        df = df.drop(columns="time")
    df["alvo"] = df["alvo"].astype(int)
    return df


def limpar_cleveland(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("ca", "thal"):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    df["alvo"] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")
    for col in ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"):
        df[col] = df[col].astype(int)
    return df


def limpar(base: NomeBase, pasta: Path | None = None) -> pd.DataFrame:
    """Base limpa e legível (útil na análise exploratória)."""
    bruto = ler_bruto(base, pasta)
    if base == "brfss":
        return limpar_brfss(bruto)
    if base == "insuficiencia":
        return limpar_insuficiencia(bruto)
    if base == "cleveland":
        return limpar_cleveland(bruto)
    raise ValueError(f"Base desconhecida: {base!r}. Use uma de {BASES}.")


# --------------------------------------------------------------------------- #
# Codificação para os modelos
# --------------------------------------------------------------------------- #
def codificar_brfss(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AgeCategory"] = _ordinal(df["AgeCategory"], ORDEM_IDADE, "AgeCategory")
    df["GenHealth"] = _ordinal(df["GenHealth"], ORDEM_SAUDE, "GenHealth")
    binarias = [
        c
        for c in df.columns
        if c != "alvo" and df[c].dtype == object and set(df[c].unique()) <= {"Yes", "No"}
    ]
    for c in binarias:
        df[c] = _sim_nao(df[c], c)
    df["Sex"] = (df["Sex"] == "Male").astype(int)
    df = df.rename(columns={"Sex": "Sex_Male"})
    multiclasse = [c for c in df.columns if df[c].dtype == object]
    return pd.get_dummies(df, columns=multiclasse, dtype=int)


def codificar_cleveland(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"], dtype=int)


def codificar(base: NomeBase, df: pd.DataFrame) -> pd.DataFrame:
    if base == "brfss":
        return codificar_brfss(df)
    if base == "cleveland":
        return codificar_cleveland(df)
    return df.copy()  # insuficiência: tudo já é numérico


def carregar(base: NomeBase, pasta: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Devolve ``(X, y)`` prontos para o pipeline: X numérico, y em 0/1.

    É a função que a Pessoa 2 (experimentos) deve usar.
    """
    df = codificar(base, limpar(base, pasta))
    y = df.pop("alvo").astype(int).rename("alvo")
    X = df.astype(float)
    return X, y


def resumo(pasta: Path | None = None) -> pd.DataFrame:
    """Tabela com linhas, variáveis, positivos e prevalência das bases disponíveis."""
    linhas = []
    for base in BASES:
        if not disponivel(base, pasta):
            continue
        X, y = carregar(base, pasta)
        linhas.append(
            {
                "base": base,
                "linhas": len(X),
                "variaveis": X.shape[1],
                "positivos": int(y.sum()),
                "prevalencia": round(float(y.mean()), 4),
            }
        )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #
def _sim_nao(s: pd.Series, nome: str) -> pd.Series:
    mapa = {"Yes": 1, "No": 0}
    fora = set(s.unique()) - set(mapa)
    if fora:
        raise ValueError(f"Valores inesperados em {nome}: {sorted(fora)}")
    return s.map(mapa).astype(int)


def _ordinal(s: pd.Series, ordem: list[str], nome: str) -> pd.Series:
    fora = set(s.unique()) - set(ordem)
    if fora:
        raise ValueError(f"Valores inesperados em {nome}: {sorted(fora)}")
    return s.map({v: i for i, v in enumerate(ordem)}).astype(int)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print(resumo())
