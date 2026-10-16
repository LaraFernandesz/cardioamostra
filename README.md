# CardioAmostra

Painel que compara, lado a lado, como técnicas de amostragem e balanceamento
(subamostragem, sobreamostragem, SMOTE, pesos de classe, ajuste de limiar...)
mudam a detecção de doença cardíaca e a calibração do modelo.

> Projeto educativo. Não tem valor diagnóstico.

## Como rodar

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash); no Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python scripts/baixar_dados.py     # baixa as 3 bases para dados/brutos/
python -m pytest
```

## Estrutura

```
src/cardioamostra/   pacote com a lógica (dados, estratégias, experimento, calibração)
app/                 painel Streamlit
scripts/             comandos (baixar dados, rodar experimentos)
notebooks/           só exploração
resultados/          .parquet gerados pelos experimentos
tests/               pytest
```

## Equipe

| Pessoa | Papel | Status |
|---|---|---|
| 1 | Dados (Fases 0 e 1) | `dados.py`, testes e notebook exploratório prontos |
| 2 | Experimentos (Fases 2 e 3) | |
| 3 | Análise e calibração (Fase 4) | |
| 4 | Painel (Fase 5) | |
| 5 | Coordenação, qualidade e comunicação (Fases 0 e 6) | |

## Como contribuir

1. Crie uma branch para a sua tarefa: `git checkout -b nome-da-tarefa`.
2. Faça os commits e envie: `git push -u origin nome-da-tarefa`.
3. Abra um pull request no GitHub e peça revisão a quem está indicado no plano.
4. Rode `python -m pytest` e `ruff check .` antes de pedir revisão.

<!-- Seção das bases: ver README_secao_dados.md (Pessoa 1), a ser incorporada aqui. -->
