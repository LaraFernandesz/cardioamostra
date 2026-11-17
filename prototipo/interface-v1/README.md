# CardioAmostra — protótipo de interface 01

## Revisão 02 — interface compacta

Solicitada pelo usuário após a primeira revisão. `compact.css` concentra os ajustes de densidade e mantém o estilo original em `style.css` para comparação. Foram reduzidos margens, intervalos, cabeçalho, altura do gráfico e espaçamento da tabela. No celular, as métricas passam a ser linhas; o resumo mantém toda a largura disponível. Os textos de apoio e as áreas de toque foram preservados.

Na configuração BRFSS / regressão logística / nenhuma correção, a altura da página caiu de 1.377 para 1.118 px em viewport de 1280 px (19%) e de 2.166 para 1.896 px em viewport de 390 px (12%). O gráfico começa 93 px antes no desktop. Conferidas larguras de 1280, 900, 390 e 320 px, títulos longos e as três telas a 320 px, sem transbordamento da página. Os botões e seletores visíveis conferidos a 900 px mantêm pelo menos 44 px de altura.

Capturas desta revisão: `desktop-compacto.jpg` e `mobile-compacto.jpg`. As capturas da primeira versão foram preservadas. Nenhuma mudança no código ou nos dados da aplicação original.

Protótipo independente em HTML, CSS e JavaScript. Nenhum arquivo da aplicação Streamlit foi alterado. Não há backend, treinamento ao vivo, publicação ou envio de dados.

## Abrir

Abra `index.html` diretamente no navegador. Os dados estão incluídos em `dados.js`; não é necessário instalar dependências. A fonte Public Sans usa Google Fonts; sem internet, a interface utiliza Arial.

Para prévia HTTP local, a partir da raiz do projeto:

```powershell
node prototipo/interface-v1/server.cjs
```

Acesse http://127.0.0.1:4173. O servidor aceita apenas conexões locais. Encerre com Ctrl+C.

## Direção visual

Laboratório editorial: marfim (#FAF9F6), tinta escura (#202C35), bordô (#9E3151), cinza esverdeado para a classe negativa e verde para melhora de métricas. Public Sans com numerais tabulares. A grade de ECG existente foi preservada na área do gráfico, sem competir com os dados. A seleção de técnicas, o gráfico, as métricas em linhas e o resumo narrativo têm composições distintas.

Referências consultadas: [Carbon Charts](https://charts.carbondesignsystem.com/) para hierarquia e leitura de gráficos e [Observable Plot](https://old.observablehq.com/plot/features/interactions) para exploração interativa. A demonstração original do projeto (`figuras/demo.gif`) foi examinada como referência visual.

## Funcionalidades

- Laboratório, Comparativo e Metodologia navegáveis, mantendo a seleção entre telas.
- Três bases, dois modelos e sete técnicas: 42 configurações reais.
- Projeções PCA das amostras originais, calculadas com `cardioamostra.apresentacao.projetar` e semente 42.
- Controle antes/depois: revela sintéticos, atenua removidos e dimensiona repetições. Os estados intermediários são animação explicativa, não treinamento parcial. As métricas representam sempre a técnica completa.
- Recall, precisão, ROC-AUC e Brier, com baseline e diferença. Comparação ordenável; exportação CSV inclui também F1 e PR-AUC.
- Estimativa em 1.000 pessoas derivada da prevalência, recall e precisão médios; explicitamente aproximada.
- Rótulos específicos para óbito na base de insuficiência cardíaca.
- Guia modal com Escape e retorno de foco; controles nativos operáveis por teclado, foco visível e suporte a movimento reduzido.
- Configuração recolhível em celular e tabela com rolagem horizontal restrita à sua região.
- Carregamento e falha de dados na inicialização. Estados demonstráveis de carregamento, vazio, erro e recuperação em Metodologia.

## Dados e reprodução

`exportar_dados.py` somente lê os resultados originais. Exporta as métricas e contagens médias dos arquivos Parquet e calcula projeções para as sete técnicas sobre cada amostra pública. Não refaz o treinamento dos modelos.

```powershell
.\.venv\Scripts\python.exe prototipo/interface-v1/exportar_dados.py
```

A prevalência utiliza as contagens médias não arredondadas do experimento. As contagens apresentadas nas barras são arredondadas, seguindo a implementação existente.

## Escopo e próximos passos

Esta versão não está integrada ao Streamlit. A exportação é local; não há upload, contas, persistência de preferências entre recarregamentos ou treinamento no navegador. A integração só deve começar após aprovação explícita.

Ainda validar com o usuário: densidade desejada, vocabulário para o público final, fluxo de exploração e necessidade de incorporar curvas de calibração à primeira tela. Testes em aparelhos físicos e leitores de tela completos permanecem pendentes.

## Verificações da versão

- Sintaxe JavaScript verificada com `node --check`.
- 42 configurações percorridas no navegador: uma técnica selecionada e quatro métricas numéricas em todas, sem erros de console.
- Ordenação por Brier, abertura de técnica pelo comparativo e geração de CSV verificadas.
- Antes/depois, teclado do controle, guia com Escape e retorno de foco, estados de interface e recuperação conferidos.
- Revisão visual em desktop e celular; larguras de 1440, 1280, 390 e 320 px usadas durante o refinamento. A largura de 320 px motivou seletores empilhados para títulos longos.
- Capturas finais em `desktop.jpg` e `mobile.jpg`.
