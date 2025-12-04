# Plano e execução de Experimentação

Esta é a entrega final do projeto de ciência de dados. O objetivo é entender a relação entre a taxa de internações por diabetes e dados demográficos gerais para entender padrões e poder auxiliar no combate aos casos graves devido à diabete.

O pré-processamento e análise exploratória dos dados (além do treino preliminar de um modelo de regressão linear) se encontra [neste link, no google colab](https://colab.research.google.com/drive/1xzG2gBIwVg-k6QGezUZigXriBMDvyAPG?usp=sharing#scrollTo=2o9TNJiv_8i8).

O arquivo `Experimento.py` gera um plano de experimentação para formular os testes com diferentes permutações de modelos e técnicas de pré-processamento e de K-Folds. A partir da tabela gerada o programa roda cada um dos 140 casos de permutações testando várias permutações de hiperparâmetros diferentes usando a técnica GridSearch - estes hiperparâmetros a serem testados estão no dicionário bayesian_search_spaces (este nome foi dado, pois, ao início utilizamos a técnica BayesianSearch para testes rápidos). No final são gerados 7780 casos diferentes. Eles são salvos cada vez que o programa é rodado em um arquivo chamado `"resultados_experimentacao_bayesian_otimizado.csv"`. No nosso caso também salvamos o arquivo com o nome `"resultados_experimentacao_longo_7k.csv"` para ele não ser reescrito quando houver uma reexecução.

Estes resultados são então analisados pelo script `Analise-Metricas.py`, que escolhe o melhor modelo e gera gráficos que são salvos como `.png` na mesma pasta. 

O relatório completo do trabalho se encontra [aqui](https://docs.google.com/document/d/1ONQZ3lKejRRVcmIDznoMtPCTM2HhT424/edit).