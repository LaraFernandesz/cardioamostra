"""Exportação local e somente leitura dos resultados para o protótipo."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cardioamostra.apresentacao import contagens, projetar
from cardioamostra.estrategias import ESTRATEGIAS

out = Path(__file__).resolve().parent
table = pd.read_parquet(ROOT / 'resultados/tabela_final.parquet')
comparison = pd.read_parquet(ROOT / 'resultados/comparacao.parquet')
counts = comparison[comparison.estrategia.eq('nenhuma') & comparison.metrica.isin(['linhas_treino', 'positivos_treino'])].groupby(['base', 'metrica']).valor.mean().unstack()
prevalences = counts.positivos_treino / counts.linhas_treino
records = table.to_dict('records')
for row in records:
    row['contagens'] = contagens(comparison, row['base'], row['modelo'], row['estrategia'])
    row['prevalencia'] = float(prevalences[row['base']])

projections = {}
for base in table.base.unique():
    sample = pd.read_parquet(ROOT / f'resultados/amostra_{base}.parquet')
    projections[base] = {'total': len(sample), 'tecnicas': {}}
    for key, strategy in ESTRATEGIAS.items():
        points = projetar(sample.drop(columns='alvo'), sample.alvo, strategy)
        # Mantém todos os pontos; posição estável entre as técnicas.
        projections[base]['tecnicas'][key] = [
            [round(float(r.pc1), 4), round(float(r.pc2), 4), r.tipo, int(r.copias)]
            for r in points.itertuples()
        ]
payload = {'resultados': records, 'projecoes': projections}
(out / 'dados.js').write_text('window.CARDIO_DATA = ' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + ';\n', encoding='utf-8')
print(f'Exportadas {len(records)} configurações e {len(projections)} bases.')
