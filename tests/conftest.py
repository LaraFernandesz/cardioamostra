"""Fixtures: pasta com versões minúsculas e falsas das 3 bases.

Permitem testar a limpeza no GitHub Actions, onde os arquivos reais não existem
(a BRFSS não deve ir para o repositório).
"""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def pasta_falsa(tmp_path: Path) -> Path:
    brfss = pd.DataFrame(
        {
            "HeartDisease": ["No", "Yes", "No", "No", "Yes"],
            "BMI": [22.0, 30.1, 25.5, 22.0, 28.0],
            "Smoking": ["Yes", "No", "No", "Yes", "Yes"],
            "AlcoholDrinking": ["No"] * 5,
            "Stroke": ["No", "Yes", "No", "No", "No"],
            "PhysicalHealth": [0.0, 30.0, 2.0, 0.0, 5.0],
            "MentalHealth": [0.0, 0.0, 5.0, 0.0, 1.0],
            "DiffWalking": ["No", "Yes", "No", "No", "Yes"],
            "Sex": ["Female", "Male", "Female", "Female", "Male"],
            "AgeCategory": ["18-24", "80 or older", "40-44", "18-24", "65-69"],
            "Race": ["White", "Black", "Hispanic", "White", "Asian"],
            "Diabetic": ["No", "Yes", "No, borderline diabetes", "No", "Yes (during pregnancy)"],
            "PhysicalActivity": ["Yes", "No", "Yes", "Yes", "No"],
            "GenHealth": ["Excellent", "Poor", "Good", "Excellent", "Fair"],
            "SleepTime": [8.0, 5.0, 7.0, 8.0, 6.0],
            "Asthma": ["No", "Yes", "No", "No", "No"],
            "KidneyDisease": ["No", "No", "No", "No", "Yes"],
            "SkinCancer": ["No", "No", "Yes", "No", "No"],
        }
    )
    # a linha 3 é duplicata exata da linha 0
    sub = tmp_path / "2020"
    sub.mkdir()
    brfss.to_csv(sub / "heart_2020_cleaned.csv", index=False)

    insuf = pd.DataFrame(
        {
            "age": [75.0, 55.0, 65.0],
            "anaemia": [0, 0, 1],
            "creatinine_phosphokinase": [582, 7861, 146],
            "diabetes": [0, 0, 0],
            "ejection_fraction": [20, 38, 20],
            "high_blood_pressure": [1, 0, 0],
            "platelets": [265000.0, 263358.03, 162000.0],
            "serum_creatinine": [1.9, 1.1, 1.3],
            "serum_sodium": [130, 136, 129],
            "sex": [1, 1, 1],
            "smoking": [0, 0, 1],
            "time": [4, 6, 7],
            "DEATH_EVENT": [1, 1, 0],
        }
    )
    insuf.to_csv(tmp_path / "heart_failure_clinical_records_dataset.csv", index=False)

    cleveland = (
        "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
        "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"
        "67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1\n"
        "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,?,3.0,0\n"
        "41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,0.0,?,0\n"
    )
    (tmp_path / "processed.cleveland.data").write_text(cleveland)
    return tmp_path
