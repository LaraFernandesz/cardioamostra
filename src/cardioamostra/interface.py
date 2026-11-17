"""Dados reais da interface: resultados agregados e projeções PCA da amostra pública."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cardioamostra.apresentacao import NOMES_BASES, contagens, projetar
from cardioamostra.estrategias import ESTRATEGIAS, MINIMO_DOENTES_SMOTE
from cardioamostra.experimento import MODELOS, ler_metadados

COMANDOS = {
    "tabela_final.parquet": "python scripts/analisar.py",
    "comparacao.parquet": "python scripts/rodar_experimentos.py --limite-linhas 50000",
    "calibracao_curvas.parquet": "python scripts/analisar.py",
}
METRICAS = ("recall", "precisao", "f1", "pr_auc", "roc_auc", "brier")


class ErroResultados(ValueError):
    """Falha recuperável que informa o arquivo e como reconstruí-lo."""

    def __init__(self, arquivo: str, motivo: str):
        self.arquivo = arquivo
        self.comando = COMANDOS.get(arquivo, "python scripts/analisar.py")
        super().__init__(f"Não foi possível ler resultados/{arquivo}: {motivo}")


def ler_tabela(pasta: Path, nome: str, colunas: set[str]) -> pd.DataFrame:
    try:
        tabela = pd.read_parquet(pasta / nome)
    except FileNotFoundError as erro:
        raise ErroResultados(nome, "arquivo não encontrado.") from erro
    except (OSError, ValueError) as erro:
        raise ErroResultados(nome, "arquivo inválido ou indisponível.") from erro
    if tabela.empty:
        raise ErroResultados(nome, "nenhum resultado disponível.")
    if not colunas.issubset(tabela.columns):
        raise ErroResultados(nome, "estrutura incompatível com o painel.")
    if tabela[list(colunas)].isna().any().any():
        raise ErroResultados(nome, "há valores ausentes nos resultados.")
    return tabela


def projecoes_da_amostra(amostra: pd.DataFrame) -> dict[str, Any]:
    """Mantém todos os pontos e registra falhas por técnica, sem inventar amostras."""
    saida: dict[str, Any] = {"total": len(amostra), "tecnicas": {}, "erros": {}}
    if amostra.empty or "alvo" not in amostra or amostra.alvo.nunique() != 2:
        saida["erro"] = "A amostra precisa conter exemplos das duas classes."
        return saida
    for chave, estrategia in ESTRATEGIAS.items():
        if chave in ("smote", "smote_tomek") and int(amostra.alvo.sum()) < MINIMO_DOENTES_SMOTE:
            saida["erros"][chave] = (
                f"Esta amostra tem menos de {MINIMO_DOENTES_SMOTE} casos positivos. "
                "Escolha outra técnica para visualizar a transformação."
            )
            continue
        try:
            pontos = projetar(amostra.drop(columns="alvo"), amostra.alvo, estrategia)
        except ValueError:
            saida["erros"][chave] = "Não foi possível projetar esta técnica na amostra disponível."
            continue
        saida["tecnicas"][chave] = [
            [round(float(p.pc1), 4), round(float(p.pc2), 4), p.tipo, int(p.copias)]
            for p in pontos.itertuples()
        ]
    return saida


def carregar_interface(pasta: Path) -> dict[str, Any]:
    """Lê a pasta de resultados da aplicação; não depende do protótipo."""
    ids = {"base", "modelo", "estrategia"}
    tabela = ler_tabela(pasta, "tabela_final.parquet", ids | set(METRICAS))
    comparacao = ler_tabela(pasta, "comparacao.parquet", ids | {"metrica", "valor"})
    if tabela.duplicated(list(ids)).any():
        raise ErroResultados("tabela_final.parquet", "há configurações duplicadas.")
    if not set(tabela.base).issubset(NOMES_BASES) or not set(tabela.modelo).issubset(MODELOS):
        raise ErroResultados("tabela_final.parquet", "base ou modelo desconhecido.")
    if not set(tabela.estrategia).issubset(ESTRATEGIAS):
        raise ErroResultados("tabela_final.parquet", "técnica desconhecida.")
    for _, grupo in tabela.groupby(["base", "modelo"]):
        if "nenhuma" not in set(grupo.estrategia):
            raise ErroResultados("tabela_final.parquet", "falta a referência sem correção.")

    registros = tabela.to_dict("records")
    for registro in registros:
        try:
            registro["contagens"] = contagens(
                comparacao, registro["base"], registro["modelo"], registro["estrategia"]
            )
            filtro = (
                comparacao.base.eq(registro["base"])
                & comparacao.modelo.eq(registro["modelo"])
                & comparacao.estrategia.eq("nenhuma")
            )
            medias = comparacao[filtro].groupby("metrica").valor.mean()
            registro["prevalencia"] = float(medias.positivos_treino / medias.linhas_treino)
        except (KeyError, AttributeError, ValueError, ZeroDivisionError) as erro:
            raise ErroResultados("comparacao.parquet", "contagens de treino incompletas.") from erro

    projecoes = {}
    for base in tabela.base.unique():
        nome = f"amostra_{base}.parquet"
        try:
            amostra = ler_tabela(pasta, nome, {"alvo"})
            projecoes[base] = projecoes_da_amostra(amostra)
        except ErroResultados:
            projecoes[base] = {
                "total": 0, "tecnicas": {}, "erros": {},
                "erro": (
                    "A amostra desta base está indisponível. As métricas continuam disponíveis."
                ),
            }

    avisos = []
    try:
        curvas = ler_tabela(
            pasta, "calibracao_curvas.parquet", ids | {"prevista", "observada", "pessoas"}
        ).to_dict("records")
    except ErroResultados:
        curvas = []
        avisos.append(
            "Curvas de calibração indisponíveis. As demais comparações continuam disponíveis."
        )
    try:
        metadados = ler_metadados(pasta / "comparacao.parquet")
    except (OSError, ValueError, KeyError):
        metadados = {}
    linhas = metadados.get("linhas_por_base", {})
    bases = {
        base: [NOMES_BASES[base], linhas.get(base)]
        for base in NOMES_BASES if base in set(tabela.base)
    }
    payload = {
        "resultados": registros, "projecoes": projecoes, "curvas": curvas,
        "bases": bases, "modelos": {k: v for k, v in MODELOS.items() if k in set(tabela.modelo)},
        "avisos": avisos,
    }
    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as erro:
        raise ErroResultados("tabela_final.parquet", "há valores numéricos inválidos.") from erro
    return payload
