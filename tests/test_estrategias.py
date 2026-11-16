import numpy as np
import pytest
from sklearn.datasets import make_classification

from cardioamostra.estrategias import ESTRATEGIAS, de_reamostragem, obter
from cardioamostra.experimento import MODELOS, montar_pipeline


@pytest.fixture(scope="module")
def dados_desbalanceados():
    return make_classification(
        n_samples=400, n_features=6, n_informative=4, weights=[0.9], random_state=42
    )


def test_registro_tem_as_7_estrategias_do_plano():
    assert list(ESTRATEGIAS) == [
        "nenhuma",
        "pesos",
        "limiar",
        "subamostragem",
        "sobreamostragem",
        "smote",
        "smote_tomek",
    ]


def test_cada_estrategia_faz_uma_coisa_so():
    for estrategia in ESTRATEGIAS.values():
        ligadas = [estrategia.reamostra, estrategia.pesos_de_classe, estrategia.ajusta_limiar]
        assert sum(ligadas) <= 1, estrategia.chave
        assert estrategia.nome and estrategia.descricao


def test_quatro_tecnicas_de_reamostragem():
    assert {e.chave for e in de_reamostragem()} == {
        "subamostragem",
        "sobreamostragem",
        "smote",
        "smote_tomek",
    }


def test_obter_chave_inexistente_explica_as_opcoes():
    with pytest.raises(KeyError, match="smote"):
        obter("adasyn")


@pytest.mark.parametrize("chave", list(ESTRATEGIAS))
@pytest.mark.parametrize("modelo", list(MODELOS))
def test_todas_as_combinacoes_treinam_e_preveem(dados_desbalanceados, chave, modelo):
    X, y = dados_desbalanceados
    pipeline = montar_pipeline(obter(chave), modelo, recall_alvo=0.7)
    pipeline.fit(X, y)
    probabilidade = pipeline.predict_proba(X)[:, 1]
    assert probabilidade.shape == (len(X),)
    assert np.all((probabilidade >= 0) & (probabilidade <= 1))
    assert set(np.unique(pipeline.predict(X))) <= {0, 1}


def test_reamostragem_iguala_as_classes_no_treino(dados_desbalanceados):
    X, y = dados_desbalanceados
    for chave in ["subamostragem", "sobreamostragem", "smote"]:
        pipeline = montar_pipeline(obter(chave), "regressao_logistica").fit(X, y)
        contador = pipeline.named_steps["contador"]
        assert contador.positivos_ * 2 == contador.linhas_, chave


def test_limiar_sem_recall_alvo_falha():
    with pytest.raises(ValueError, match="recall_alvo"):
        montar_pipeline(obter("limiar"), "regressao_logistica")
