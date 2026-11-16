"""Registro das 7 estratégias para lidar com o desbalanceamento de classes.

Cada estratégia é um objeto com o mesmo formato. Para acrescentar uma técnica
nova (ADASYN, Borderline-SMOTE...), basta incluir uma linha em ``_REGISTRO``;
o experimento e o seletor do painel se montam a partir daqui.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

VIZINHOS_SMOTE = 5
MINIMO_DOENTES_SMOTE = VIZINHOS_SMOTE + 1


@dataclass(frozen=True)
class Estrategia:
    """Uma forma de tratar o desbalanceamento.

    ``criar_amostrador`` recebe a semente e devolve o passo de reamostragem do
    pipeline (ou é ``None`` quando a estratégia não reamostra).
    """

    chave: str
    nome: str
    descricao: str
    criar_amostrador: Callable[[int], Any] | None = None
    pesos_de_classe: bool = False
    ajusta_limiar: bool = False

    @property
    def reamostra(self) -> bool:
        return self.criar_amostrador is not None


_REGISTRO: tuple[Estrategia, ...] = (
    Estrategia(
        chave="nenhuma",
        nome="Nenhuma correção",
        descricao="Treina com os dados como estão e decide com limiar 0,5.",
    ),
    Estrategia(
        chave="pesos",
        nome="Pesos de classe",
        descricao="Erros na classe minoritária pesam mais no treino; nenhuma linha é criada.",
        pesos_de_classe=True,
    ),
    Estrategia(
        chave="limiar",
        nome="Ajuste de limiar",
        descricao=(
            "Treina sem correção e escolhe, só no treino, o limiar que alcança o recall "
            "da melhor técnica de reamostragem. As probabilidades não mudam."
        ),
        ajusta_limiar=True,
    ),
    Estrategia(
        chave="subamostragem",
        nome="Subamostragem aleatória",
        descricao="Descarta pessoas sem doença ao acaso até igualar as classes.",
        criar_amostrador=lambda semente: RandomUnderSampler(random_state=semente),
    ),
    Estrategia(
        chave="sobreamostragem",
        nome="Sobreamostragem aleatória",
        descricao="Repete doentes ao acaso até igualar as classes.",
        criar_amostrador=lambda semente: RandomOverSampler(random_state=semente),
    ),
    Estrategia(
        chave="smote",
        nome="SMOTE",
        descricao="Cria pacientes sintéticos sobre a linha entre dois doentes vizinhos.",
        criar_amostrador=lambda semente: SMOTE(k_neighbors=VIZINHOS_SMOTE, random_state=semente),
    ),
    Estrategia(
        chave="smote_tomek",
        nome="SMOTE + Tomek",
        descricao="SMOTE seguido da remoção dos pares de vizinhos de classes opostas (Tomek).",
        criar_amostrador=lambda semente: SMOTETomek(
            smote=SMOTE(k_neighbors=VIZINHOS_SMOTE, random_state=semente), random_state=semente
        ),
    ),
)

ESTRATEGIAS: dict[str, Estrategia] = {e.chave: e for e in _REGISTRO}


def obter(chave: str) -> Estrategia:
    """Devolve a estratégia pela chave, com mensagem clara se ela não existir."""
    try:
        return ESTRATEGIAS[chave]
    except KeyError as erro:
        validas = ", ".join(ESTRATEGIAS)
        raise KeyError(f"Estratégia '{chave}' não existe. Use uma de: {validas}") from erro


def de_reamostragem() -> list[Estrategia]:
    """Estratégias que criam ou removem linhas do treino."""
    return [e for e in _REGISTRO if e.reamostra]
