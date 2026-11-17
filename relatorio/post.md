<!-- Texto do post (LinkedIn ou blog). Anexar figuras/recall_brier.png ou figuras/demo.gif. -->

SMOTE encontra mais doentes. Mas a que custo?

No CardioAmostra, nosso projeto em equipe, testamos 7 formas de lidar com dados desbalanceados em 3 bases públicas de doença cardíaca. Validação cruzada 5 × 3, com a reamostragem só no treino.

Na BRFSS 2020, onde só 9% das pessoas têm doença cardíaca:

→ Todas as técnicas de reamostragem e os pesos de classe levam o recall de 0,10 para 0,78.
→ Mas o modelo passa a prever cerca de 4 vezes o risco real. O Brier vai de 0,069 para 0,170.
→ A ROC-AUC não melhora com nenhuma delas.
→ Ajustar o limiar de decisão dá o mesmo recall de 0,78, e as probabilidades continuam corretas.

E um achado que não esperávamos: SMOTE com gradient boosting quase não funciona nessa base (recall de 0,16). O SMOTE cria pacientes com valores impossíveis, como 7,3 dias de saúde ruim, e as árvores aprendem a reconhecer quem é sintético. Arredondando, o recall volta a 0,71.

Para um modelo de risco, a probabilidade importa tanto quanto acertar quem é doente. Antes de reamostrar, vale testar o limiar.

O painel mostra tudo lado a lado, técnica por técnica: https://cardioamostra-8h6d8kmb9qysg4yrafgyz9.streamlit.app/
Código, relatório e dados: https://github.com/LaraFernandesz/cardioamostra

Projeto educativo, sem valor diagnóstico.

#MachineLearning #CiênciaDeDados #Saúde #Python
