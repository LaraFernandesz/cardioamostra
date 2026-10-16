# Bases brutas

Os arquivos desta pasta **não vão para o GitHub** (estão no `.gitignore`).
Rode `python scripts/baixar_dados.py` ou baixe manualmente e coloque aqui
(subpastas são aceitas; o código procura pelo nome do arquivo):

| Base | Arquivo esperado | Onde baixar |
|---|---|---|
| BRFSS 2020 (Kaggle) | `heart_2020_cleaned.csv` | https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease (o zip traz a pasta `2020/`) |
| Heart Failure Clinical Records (UCI 519) | `heart_failure_clinical_records_dataset.csv` | https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records |
| Heart Disease – Cleveland (UCI 45) | `processed.cleveland.data` | https://archive.ics.uci.edu/dataset/45/heart+disease |

Confira com `python -m pytest`: os 3 testes `test_contagens_reais` devem passar.
