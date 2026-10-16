"""Testes de dados.py.

* Testes de lógica: usam bases falsas e rodam sempre (inclusive no CI).
* Testes de contagem: usam as bases reais e são pulados se os arquivos
  não estiverem em dados/brutos/.
"""

from pathlib import Path

import pytest

from cardioamostra import dados
from cardioamostra.dados import BASES, ESPERADO, carregar, disponivel, limpar


# --------------------------------------------------------------- lógica ----
@pytest.mark.parametrize("base", BASES)
def test_alvo_binario_e_x_numerico(pasta_falsa: Path, base: str) -> None:
    X, y = carregar(base, pasta_falsa)
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)
    assert "alvo" not in X.columns
    assert X.isna().sum().sum() == 0
    assert all(dt.kind == "f" for dt in X.dtypes)


def test_brfss_remove_duplicatas(pasta_falsa: Path) -> None:
    assert len(limpar("brfss", pasta_falsa)) == 4  # 5 linhas, 1 duplicata


def test_brfss_ordinais(pasta_falsa: Path) -> None:
    X, _ = carregar("brfss", pasta_falsa)
    assert X["AgeCategory"].min() == 0 and X["AgeCategory"].max() == 12
    assert set(X["GenHealth"]) == {0, 1, 2, 4}  # Poor, Fair, Good, Excellent
    assert "Sex_Male" in X.columns
    assert any(c.startswith("Race_") for c in X.columns)
    assert any(c.startswith("Diabetic_") for c in X.columns)


def test_brfss_valor_inesperado_falha(pasta_falsa: Path) -> None:
    arq = dados.caminho("brfss", pasta_falsa)
    texto = arq.read_text().replace("Excellent", "Excelente", 1)
    arq.write_text(texto)
    with pytest.raises(ValueError, match="GenHealth"):
        carregar("brfss", pasta_falsa)


def test_insuficiencia_remove_tempo(pasta_falsa: Path) -> None:
    X, y = carregar("insuficiencia", pasta_falsa)
    assert "time" not in X.columns
    assert y.tolist() == [1, 1, 0]


def test_cleveland_alvo_e_ausentes(pasta_falsa: Path) -> None:
    X, y = carregar("cleveland", pasta_falsa)
    assert y.tolist() == [0, 1, 1, 0, 0]  # num > 0
    assert len(X) == 5  # nenhuma linha perdida pelos ausentes
    assert any(c.startswith("thal_") for c in X.columns)


def test_arquivo_ausente_explica_o_que_fazer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="LEIA-ME"):
        carregar("brfss", tmp_path)


# ---------------------------------------------------- contagens reais ----
@pytest.mark.parametrize("base", BASES)
def test_contagens_reais(base: str) -> None:
    if not disponivel(base):
        pytest.skip(f"arquivo da base {base} não está em dados/brutos/")
    esperado = ESPERADO[base]
    assert len(dados.ler_bruto(base)) == esperado["linhas_brutas"]
    X, y = carregar(base)
    assert len(X) == esperado["linhas"]
    assert y.mean() == pytest.approx(esperado["prevalencia"], abs=0.0005)
