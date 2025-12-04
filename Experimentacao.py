import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
import gc
import warnings
import time
from itertools import product



# Definir todas as combinações de cenários
tecnicas = [
    'Regressão Linear',
    'Árvore de Decisão',
    'Random Forest',
    'XGBoost',
    'K-NN',
    'Regressão Linear Bayesiana',
    'Redes Neurais'
]

folds = [3, 4, 5, 6, 7]
dataframes = [
    ('Com Clustering', 'Reduzido'),
    ('Sem Clustering', 'Reduzido'),
    ('Com Clustering', 'Não Reduzido'),
    ('Sem Clustering', 'Não Reduzido')
]

# Criar lista para armazenar as linhas do DataFrame
rows = []

for df_type in dataframes:
    for fold in folds:
        for tec in tecnicas:
            row = {
                'Regressão Linear': 'X' if tec == 'Regressão Linear' else '',
                # 'Regressão Polinomial': 'X' if tec == 'Regressão Polinomial' else '',
                'Árvore de Decisão': 'X' if tec == 'Árvore de Decisão' else '',
                'Random Forest': 'X' if tec == 'Random Forest' else '',
                'XGBoost': 'X' if tec == 'XGBoost' else '',
                'K-NN': 'X' if tec == 'K-NN' else '',
                'Regressão Linear Bayesiana': 'X' if tec == 'Regressão Linear Bayesiana' else '',
                'Redes Neurais': 'X' if tec == 'Redes Neurais' else '',
                '3-Fold': 'X' if fold == 3 else '',
                '4-Fold': 'X' if fold == 4 else '',
                '5-Fold': 'X' if fold == 5 else '',
                '6-Fold': 'X' if fold == 6 else '',
                '7-Fold': 'X' if fold == 7 else '',
                'Reduzido': 'X' if df_type[1] == 'Reduzido' else '',
                'Clustering': 'X' if 'Com Clustering' in df_type else '',
                'R²': '',
                'MSE': ''
            }
            rows.append(row)

# Criar DataFrame
df_experimentos = pd.DataFrame(rows)

# Reorganizar as colunas para melhor visualização
col_order = [
    'Reduzido', 'Clustering', 'Regressão Linear', #'Regressão Polinomial',
    'Árvore de Decisão', 'Random Forest', 'XGBoost', 'K-NN',
    'Regressão Linear Bayesiana', 'Redes Neurais', '3-Fold', '4-Fold',
    '5-Fold', '6-Fold', '7-Fold', 'R²', 'MSE'
]
df_experimentos = df_experimentos[col_order]

# Exibir o DataFrame
print(df_experimentos)


# Carregar os dataframes
print("Carregando dados...")
df_merged = pd.read_csv('df_merged.csv')
df_merged_reduzido = pd.read_csv('df_merged_reduzido.csv')
df_clusters = pd.read_csv('df_clusters.csv')
df_clusters_reduzido = pd.read_csv('df_clusters_reduzido.csv')

# Limpar arquivo de resultados anterior, se existir
open('resultados_experimentacao_bayesian_otimizado.csv', 'w').close()


# Reorganizar as colunas para melhor visualização
col_order = [
    'Reduzido', 'Clustering', 'Regressão Linear', #'Regressão Polinomial',
    'Árvore de Decisão', 'Random Forest', 'XGBoost', 'K-NN',
    'Regressão Linear Bayesiana', 'Redes Neurais', '3-Fold', 'R²', 'MSE'
]
df_experimentos = df_experimentos[col_order]

# Função para selecionar o dataframe correto
def selecionar_dataframe(reduzido, clustering):
    if clustering == 'X' and reduzido == 'X':
        return df_clusters_reduzido
    elif clustering == 'X' and reduzido != 'X':
        return df_clusters
    elif clustering != 'X' and reduzido == 'X':
        return df_merged_reduzido
    else:
        return df_merged

def extrair_n_folds(experimento):
    if experimento.get('3-Fold') == 'X':
        return 3
    elif experimento.get('4-Fold') == 'X':
        return 4
    elif experimento.get('5-Fold') == 'X':
        return 5
    elif experimento.get('6-Fold') == 'X':
        return 6
    elif experimento.get('7-Fold') == 'X':
        return 7
    else:
        return 3 

# Uma versão mais leve dos espaços de busca para economizar memória

# Definir espaços de busca mais conservadores para economizar memória
# bayesian_search_spaces = {
#     'Regressão Linear': {},
#     'Árvore de Decisão': {
#         'max_depth': [3, 8],  # Reduzido
#         'min_samples_split': [2, 5],  # Reduzido
#     },
#     'Random Forest': {
#         'n_estimators': [30, 80],  # Reduzido significativamente
#         'max_depth': [3, 8],  # Reduzido
#     },
#     'XGBoost': {
#         'n_estimators': [20, 40],  # Reduzido significativamente
#         'max_depth': [3, 4],  # Reduzido
#         'learning_rate': [0.05, 0.2],  # Reduzido
#     },
#     'K-NN': {
#         'n_neighbors': [3, 10],  # Reduzido
#         'weights': ['uniform', 'distance'],
#     },
#     'Regressão Linear Bayesiana': {
#         'alpha_1': [1e-6, 1e-4],  # Reduzido
#         'alpha_2': [1e-6, 1e-4],  # Reduzido
#     },
#     'Redes Neurais': {
#         # 'learning_rate_init': Real(0.001, 0.1, prior='log-uniform'),
#         'hidden_layer_sizes': [(50,), (30,), (50, 30)],
#         'activation': ['relu', 'tanh'],
#         'alpha': [0.0001, 0.001, 0.01],
#     }
# }

bayesian_search_spaces = {

    'Regressão Linear': {},

    'Árvore de Decisão': {
        'max_depth': [3, 5, 8, 12, 16],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2']
    },

    'Random Forest': {
        'n_estimators': [30, 80], 
        'max_depth': [3, 8], 
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 5],
    },

    'XGBoost': {
        'n_estimators': [20, 40],  
        'max_depth': [3, 4],  
        'learning_rate': [0.05, 0.2],
        'subsample': [0.6, 0.8, 1.0],
    },

    'K-NN': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },

    'Regressão Linear Bayesiana': {
        'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
        'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4]
    },

    'Redes Neurais': {
        'hidden_layer_sizes': [
            (20,), (50,),
        ],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01],
        'learning_rate_init': [0.001]
    }
}


# Lista para armazenar todos os resultados
resultados_expandidos = []

# Total de experimentos
total_experimentos = len(df_experimentos)
experimentos_completados = 0

def executar_busca_e_avaliar(
    nome_tecnica,
    modelo,
    param_space,
    X_train, X_test,
    y_train, y_test,
    experimento,
    resultados_expandidos,
    n_folds=3,
    n_iter=8,
    tipo_busca="bayes"
):
    # ✅ Caso especial: modelo sem hiperparâmetros (ex: Regressão Linear)
    if not param_space:
        print(f"⚠️ {nome_tecnica} não possui hiperparâmetros. Treinando modelo único.")

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        nova_linha = experimento.copy()
        nova_linha['Modelo'] = nome_tecnica
        nova_linha['Tipo de Busca'] = 'nenhuma'
        nova_linha['R²'] = float(r2)
        nova_linha['MSE'] = float(mse)
        nova_linha['Tempo Execução (s)'] = 0.0

        resultados_expandidos.append(nova_linha)

        del y_pred
        gc.collect()

        print(f"✅ {nome_tecnica} finalizado (modelo único)")
        return
    try:
        inicio = time.time()

        # ✅ Criar objeto de busca
        if tipo_busca == "grid":
            busca = GridSearchCV(
                estimator=model,
                param_grid=param_space,
                cv=n_folds,
                scoring='r2',
                n_jobs=1,
                verbose=0,
                return_train_score=False
            )
        else:
            busca = BayesSearchCV(
                estimator=model,
                search_spaces=param_space,
                cv=n_folds,
                scoring='r2',
                n_iter=n_iter,
                random_state=42,
                n_jobs=1,
                verbose=0,
                return_train_score=False
            )

        busca.fit(X_train, y_train)

        fim = time.time()
        tempo_real = fim - inicio

        print(f"✅ {nome_tecnica} finalizado em {tempo_real/60:.2f} min")

        resultados_cv = busca.cv_results_

        for i in range(len(resultados_cv['params'])):
            params = resultados_cv['params'][i]

            # ✅ Corrigir automaticamente parâmetros que vêm como array
            params_corrigidos = {}
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    if v.size == 1:
                        # Array de um único elemento
                        params_corrigidos[k] = v.item()
                    elif v.dtype == object and len(v) == 1 and isinstance(v[0], tuple):
                        # Caso especial para hidden_layer_sizes: array([(50, 30)])
                        params_corrigidos[k] = v[0]
                    else:
                        # Outros casos - converter para lista
                        params_corrigidos[k] = v.tolist()
                elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], tuple):
                    # Lista com uma tupla: [(50, 30)]
                    params_corrigidos[k] = v[0]
                else:
                    params_corrigidos[k] = v

            modelo_temp = modelo.__class__(**params_corrigidos)

            modelo_temp.fit(X_train, y_train)

            y_pred = modelo_temp.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            nova_linha = experimento.copy()
            nova_linha['Modelo'] = nome_tecnica
            nova_linha['Tipo de Busca'] = tipo_busca
            nova_linha['Tempo Execução (s)'] = round(tempo_real, 2)

            # ✅ Garantir que R² e MSE sejam escalares
            if isinstance(r2, (list, np.ndarray)):
                r2_val = float(np.mean(r2))
            else:
                r2_val = float(r2)

            if isinstance(mse, (list, np.ndarray)):
                mse_val = float(np.mean(mse))
            else:
                mse_val = float(mse)

            nova_linha['R²'] = r2_val
            nova_linha['MSE'] = mse_val


            for p, v in params.items():
                nova_linha[p] = v

            resultados_expandidos.append(nova_linha)

            del modelo_temp, y_pred
            gc.collect()

        busca.cv_results_ = None
        del busca, resultados_cv
        gc.collect()

    except Exception as e:
        print(f"❌ FALHA NA FUNÇÃO executar_busca_e_avaliar → Modelo: {nome_tecnica}")
        print(f"❌ Erro real: {type(e).__name__}: {e}")

        nova_linha = experimento.copy()
        nova_linha['Modelo'] = nome_tecnica
        nova_linha['R²'] = np.nan
        nova_linha['MSE'] = np.nan
        nova_linha['Erro'] = str(e)
        resultados_expandidos.append(nova_linha)

        gc.collect()




print("Iniciando execução do plano de experimentação com Bayesian Search otimizado...")

for idx, experimento in df_experimentos.iterrows():
    # Determinar qual técnica está sendo usada
    tecnica = None
    for tec in tecnicas:
        if experimento[tec] == 'X':
            tecnica = tec
            break

    if tecnica is None:
        continue

    df = selecionar_dataframe(experimento['Reduzido'], experimento['Clustering'])

    X = df.drop(columns=['Taxa a cada 100 mil hab.', 'id_municipio'])
    y = df['Taxa a cada 100 mil hab.']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Determinar o número de folds
    n_folds = extrair_n_folds(experimento)
    if n_folds > len(X_train):
        print(f"⚠️ Fold {n_folds} maior que treino {len(X_train)}. Ajustando para 3.")
        n_folds = 3

    try:
        # Configurar o modelo e bayesian search com configurações otimizadas
        n_iter = 8 

        if tecnica == 'Regressão Linear':
            model = LinearRegression()
            executar_busca_e_avaliar(
                nome_tecnica=tecnica,
                modelo=model,
                param_space=bayesian_search_spaces[tecnica],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experimento=experimento,
                resultados_expandidos=resultados_expandidos,
                n_folds=n_folds,
                n_iter=8,
                tipo_busca="grid"
            )

        elif tecnica == 'Árvore de Decisão':
            model = DecisionTreeRegressor(random_state=42)

            executar_busca_e_avaliar(
                nome_tecnica=tecnica,
                modelo=model,
                param_space=bayesian_search_spaces[tecnica],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experimento=experimento,
                resultados_expandidos=resultados_expandidos,
                n_folds=n_folds,
                n_iter=8,
                tipo_busca="grid"
            )

        elif tecnica == 'Random Forest':
            model = RandomForestRegressor(random_state=42, n_jobs=1)  # n_jobs=1 para RF também
            executar_busca_e_avaliar(
                nome_tecnica=tecnica,
                modelo=model,
                param_space=bayesian_search_spaces[tecnica],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experimento=experimento,
                resultados_expandidos=resultados_expandidos,
                n_folds=n_folds,
                n_iter=8,
                tipo_busca="grid"
            )

        elif tecnica == 'XGBoost':
            model = xgb.XGBRegressor(random_state=42, n_jobs=1)  # n_jobs=1 para XGBoost também
            executar_busca_e_avaliar(
                nome_tecnica=tecnica,
                modelo=model,
                param_space=bayesian_search_spaces[tecnica],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experimento=experimento,
                resultados_expandidos=resultados_expandidos,
                n_folds=n_folds,
                n_iter=8,
                tipo_busca="grid"
            )

        elif tecnica == 'K-NN':
            model = KNeighborsRegressor(n_jobs=1)  # n_jobs=1 para K-NN
            
            executar_busca_e_avaliar(
                nome_tecnica=tecnica,
                modelo=model,
                param_space=bayesian_search_spaces[tecnica],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experimento=experimento,
                resultados_expandidos=resultados_expandidos,
                n_folds=n_folds,
                n_iter=8,
                tipo_busca="grid"
            )

        elif tecnica == 'Regressão Linear Bayesiana':
            model = BayesianRidge()
            executar_busca_e_avaliar(
                nome_tecnica=tecnica,
                modelo=model,
                param_space=bayesian_search_spaces[tecnica],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experimento=experimento,
                resultados_expandidos=resultados_expandidos,
                n_folds=n_folds,
                n_iter=8,
                tipo_busca="grid"
            )

        elif tecnica == 'Redes Neurais':
            print(f"  Executando Redes Neurais para experimento {idx} com GridSearch (sem scaler)...")
            model = MLPRegressor(
                random_state=42,
                max_iter=800,   # limite seguro
                early_stopping=True,  # converge mais rápido
                n_iter_no_change=20,
                tol=1e-4
            )
            executar_busca_e_avaliar(
                nome_tecnica=tecnica,
                modelo=model,
                param_space=bayesian_search_spaces[tecnica],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experimento=experimento,
                resultados_expandidos=resultados_expandidos,
                n_folds=n_folds,
                n_iter=8,
                tipo_busca="grid"
            )
            print(f"✅ Retorno da função para Redes Neurais no experimento {idx}")

    except Exception as e:
        print(f"Erro no experimento {idx} ({tecnica}): {str(e)}")
        # Adicionar linha com erro
        nova_linha = experimento.copy()
        nova_linha['R²'] = np.nan
        nova_linha['MSE'] = np.nan
        nova_linha['Erro'] = str(e)
        resultados_expandidos.append(nova_linha)

    # Limpar memória após cada experimento
    del df, X, y
    gc.collect()

    experimentos_completados += 1
    progresso = (experimentos_completados / total_experimentos) * 100
    print(f"Progresso: {progresso:.2f}% ({experimentos_completados}/{total_experimentos}) - Último: {tecnica}")

# Criar DataFrame final com todos os resultados
df_resultados_final = pd.DataFrame(resultados_expandidos)

# Preencher NaN para hiperparâmetros que não se aplicam a certos modelos
df_resultados_final = df_resultados_final.fillna('N/A')

print("Experimentação com Bayesian Search otimizada concluída!")
print(f"Total de combinações testadas: {len(df_resultados_final)}")

# Salvar resultados em CSV
df_resultados_final.to_csv('resultados_experimentacao_bayesian_otimizado.csv', index=False)


print("\nResumo dos melhores resultados por técnica:")
for tecnica in tecnicas:
    tech_results = df_resultados_final[df_resultados_final[tecnica] == 'X']
    if len(tech_results) > 0:
        # Converter R² para numérico, forçando erros para NaN
        tech_results['R²_numeric'] = pd.to_numeric(tech_results['R²'], errors='coerce')
        
        # Filtrar apenas valores numéricos válidos
        valid_results = tech_results[tech_results['R²_numeric'].notna()]
        
        if len(valid_results) > 0:
            best_r2_idx = valid_results['R²_numeric'].idxmax()
            best_r2_row = valid_results.loc[best_r2_idx]
            best_r2 = best_r2_row['R²_numeric']
            
            # Verificar se é um número
            if pd.notna(best_r2):
                print(f"{tecnica} - Melhor R²: {best_r2:.4f}")
            else:
                print(f"{tecnica} - Sem resultados válidos")
        else:
            print(f"{tecnica} - Sem resultados válidos")

# Mostrar as primeiras linhas dos resultados
print("\nPrimeiras linhas dos resultados:")
print(df_resultados_final.head(10))
executar_busca_e_avaliar
