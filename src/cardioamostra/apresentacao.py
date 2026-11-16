"""Dados prontos para o painel: rankings, comparações, contagens e projeção PCA.

Fica no pacote (e não em ``app/``) para ser testado sem abrir o navegador. O
painel só chama estas funções sobre os arquivos de ``resultados/``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cardioamostra.estrategias import ESTRATEGIAS, Estrategia, de_reamostragem

NOMES_BASES = {
    "brfss": "BRFSS 2020",
    "insuficiencia": "Insuficiência cardíaca",
    "cleveland": "Cleveland",
}
NOMES_METRICAS = {
    "recall": "Recall",
    "precisao": "Precisão",
    "f1": "F1",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "brier": "Brier",
    "razao_previsto_observado": "Previsto ÷ observado",
}
EXPLICACOES = {
    "recall": "Doentes encontrados",
    "precisao": "Alertas que eram doença",
    "f1": "Equilíbrio entre recall e precisão",
    "pr_auc": "Ordenação focada nos doentes",
    "roc_auc": "Capacidade de ordenar",
    "brier": "Erro das probabilidades",
}
MENOR_MELHOR = {"brier"}

TIPOS_PONTO = ("Sem doença", "Com doença (real)", "Sintético", "Descartado")


# --------------------------------------------------------------------------- #
# Tabelas e comparações
# --------------------------------------------------------------------------- #
def filtrar(tabela: pd.DataFrame, base: str, modelo: str) -> pd.DataFrame:
    """Linhas de uma base e um modelo, indexadas pela estratégia."""
    filtro = (tabela["base"] == base) & (tabela["modelo"] == modelo)
    return tabela[filtro].set_index("estrategia")


def ranking(tabela: pd.DataFrame, base: str, modelo: str, metrica: str) -> pd.DataFrame:
    """As 7 estratégias ordenadas da melhor para a pior na métrica escolhida."""
    t = filtrar(tabela, base, modelo)
    t = t.sort_values(metrica, ascending=metrica in MENOR_MELHOR, kind="stable")
    t.insert(0, "tecnica", [ESTRATEGIAS[e].nome for e in t.index])
    return t


def comparar_com_nenhuma(
    tabela: pd.DataFrame, base: str, modelo: str, estrategia: str
) -> dict[str, dict[str, float]]:
    """Valor de cada métrica e a diferença em relação a "nenhuma correção"."""
    t = filtrar(tabela, base, modelo)
    saida = {}
    for metrica in EXPLICACOES:
        valor = float(t.loc[estrategia, metrica])
        saida[metrica] = {"valor": valor, "delta": valor - float(t.loc["nenhuma", metrica])}
    return saida


def contagens(comparacao: pd.DataFrame, base: str, modelo: str, estrategia: str) -> dict[str, int]:
    """Tamanho médio do treino antes e depois da técnica (barra "Antes / Depois")."""
    filtro = (
        (comparacao["base"] == base)
        & (comparacao["modelo"] == modelo)
        & (comparacao["estrategia"] == estrategia)
    )
    medias = comparacao[filtro].groupby("metrica")["valor"].mean()
    return {
        "linhas_antes": int(round(medias["linhas_treino"])),
        "doentes_antes": int(round(medias["positivos_treino"])),
        "linhas_depois": int(round(medias["linhas_apos"])),
        "doentes_depois": int(round(medias["positivos_apos"])),
    }


def em_mil_pessoas(prevalencia: float, recall: float, precisao: float) -> dict[str, int]:
    """Matriz de confusão aproximada para 1.000 pessoas do teste.

    Deriva das médias de recall e precisão, por isso é aproximada (≈).
    """
    doentes = 1000 * prevalencia
    encontrados = recall * doentes
    alarmes = encontrados / precisao - encontrados if precisao > 0 else 0.0
    return {
        "doentes_encontrados": int(round(encontrados)),
        "doentes_perdidos": int(round(doentes - encontrados)),
        "alarmes_falsos": int(round(alarmes)),
        "saudaveis_corretos": int(round(1000 - doentes - alarmes)),
    }


def resumo_comparativo(tabela: pd.DataFrame, base: str, modelo: str) -> str:
    """Frase com os números que contam a conclusão da tela Comparativo."""
    t = filtrar(tabela, base, modelo)
    corrigem = [e.chave for e in de_reamostragem()] + ["pesos"]
    nenhuma, limiar, outras = t.loc["nenhuma"], t.loc["limiar"], t.loc[corrigem]

    def faixa(coluna: str, casas: int) -> str:
        menor, maior = outras[coluna].min(), outras[coluna].max()
        if round(menor, casas) == round(maior, casas):
            return _num(menor, casas)
        return f"{_num(menor, casas)} a {_num(maior, casas)}"

    return (
        f"Sem correção, o modelo encontra {_num(nenhuma['recall'], 2)} dos doentes, com Brier de "
        f"{_num(nenhuma['brier'], 3)}. As técnicas de reamostragem e os pesos de classe levam o "
        f"recall a {faixa('recall', 2)}, e o Brier fica em {faixa('brier', 3)}. "
        f"O ajuste de limiar chega a recall de {_num(limiar['recall'], 2)} com Brier de "
        f"{_num(limiar['brier'], 3)}, porque não mexe nas probabilidades."
    )


# --------------------------------------------------------------------------- #
# Projeção PCA da amostra (calculada ao vivo no painel)
# --------------------------------------------------------------------------- #
def projetar(
    X: pd.DataFrame, y: pd.Series, estrategia: Estrategia, semente: int = 42
) -> pd.DataFrame:
    """Aplica a técnica na amostra e projeta tudo em 2 dimensões.

    O PCA é ajustado só nos pontos originais, para os eixos não mudarem entre
    técnicas. Cada ponto original diz se foi descartado e quantas cópias tem
    depois da técnica; os pontos novos são marcados como sintéticos.
    """
    escala = StandardScaler().fit(X)
    original = escala.transform(X)
    y_arr = np.asarray(y).astype(int)
    pca = PCA(n_components=2, random_state=semente).fit(original)

    if estrategia.criar_amostrador is None:
        novo, y_novo = original, y_arr
    else:
        novo, y_novo = estrategia.criar_amostrador(semente).fit_resample(original, y_arr)

    chaves_originais = [_chave(linha) for linha in original]
    copias = pd.Series([_chave(linha) for linha in novo]).value_counts()
    conhecidas = set(chaves_originais)
    eh_sintetico = np.array([_chave(linha) not in conhecidas for linha in novo])

    pontos_originais = pd.DataFrame(pca.transform(original), columns=["pc1", "pc2"])
    pontos_originais["copias"] = [int(copias.get(k, 0)) for k in chaves_originais]
    pontos_originais["tipo"] = np.where(y_arr == 1, TIPOS_PONTO[1], TIPOS_PONTO[0])
    pontos_originais.loc[pontos_originais["copias"] == 0, "tipo"] = TIPOS_PONTO[3]

    if not eh_sintetico.any():
        return pontos_originais
    if (y_novo[eh_sintetico] != 1).any():
        raise AssertionError("A técnica criou pontos sintéticos da classe sem doença.")
    sinteticos = pd.DataFrame(pca.transform(novo[eh_sintetico]), columns=["pc1", "pc2"])
    sinteticos["copias"] = 1
    sinteticos["tipo"] = TIPOS_PONTO[2]
    return pd.concat([pontos_originais, sinteticos], ignore_index=True)


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #
def _chave(linha: Any) -> bytes:
    # + 0.0 transforma -0.0 em 0.0, que senão viraria uma chave diferente
    return (np.round(np.asarray(linha, dtype=float), 9) + 0.0).tobytes()


def _num(valor: float, casas: int) -> str:
    """Número no formato brasileiro (vírgula decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",")
