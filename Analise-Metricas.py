import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')

# Configurações estéticas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def format_scientific(x, pos):
    """Formatador para notação científica mais legível"""
    if x == 0:
        return '0'
    if abs(x) < 0.0001:
        return f'{x:.2e}'
    elif abs(x) < 0.01:
        return f'{x:.6f}'
    else:
        return f'{x:.4f}'

def load_and_prepare_data(filepath):
    """Carrega e prepara os dados do CSV"""
    df = pd.read_csv(filepath)
    
    # Identificar qual técnica foi usada em cada linha
    tecnicas = [
        'Regressão Linear', 'Árvore de Decisão', 'Random Forest', 
        'XGBoost', 'K-NN', 'Regressão Linear Bayesiana', 'Redes Neurais'
    ]
    
    # Criar coluna 'Modelo' baseada nas colunas de técnicas
    def get_model_name(row):
        for tecnica in tecnicas:
            if pd.notna(row.get(tecnica, '')) and row[tecnica] != '':
                return tecnica
        return 'Desconhecido'
    
    df['Modelo'] = df.apply(get_model_name, axis=1)
    
    # Converter colunas numéricas
    numeric_cols = ['R²', 'MSE']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Criar identificador único para cada configuração
    df['Config_ID'] = df.groupby(['Modelo', 'Reduzido', 'Clustering']).cumcount()
    
    return df, tecnicas

def plot_model_comparison(df, tecnicas):
    """Plota comparação entre diferentes tipos de modelos"""
    
    # Encontrar o melhor resultado para cada modelo
    best_by_model = []
    for modelo in tecnicas:
        model_df = df[df['Modelo'] == modelo]
        if len(model_df) > 0:
            best_idx = model_df['R²'].idxmax() if not model_df['R²'].isna().all() else None
            if best_idx is not None:
                best_by_model.append(model_df.loc[best_idx])
    
    if not best_by_model:
        print("Nenhum dado válido encontrado para comparação de modelos.")
        return
    
    best_df = pd.DataFrame(best_by_model)
    
    # Gráfico 1: Comparação de R²
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Ordenar por R²
    best_df_sorted = best_df.sort_values('R²', ascending=False)
    
    # Plot R²
    bars1 = axes[0].bar(range(len(best_df_sorted)), best_df_sorted['R²'], 
                       color=plt.cm.Set3(np.arange(len(best_df_sorted))))
    
    axes[0].set_title('Comparação do Melhor R² por Tipo de Modelo', fontsize=18, fontweight='bold', pad=20)
    axes[0].set_xlabel('Modelo', fontsize=14)
    axes[0].set_ylabel('R² Score', fontsize=14)
    axes[0].set_xticks(range(len(best_df_sorted)))
    axes[0].set_xticklabels(best_df_sorted['Modelo'], rotation=45, ha='right', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Adicionar valores nas barras
    for i, (bar, r2) in enumerate(zip(bars1, best_df_sorted['R²'])):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Gráfico 2: Comparação de MSE
    best_df_sorted_mse = best_df.sort_values('MSE')
    
    # Plot MSE (em escala log para melhor visualização se necessário)
    mse_values = best_df_sorted_mse['MSE'].values
    use_log = any(mse_values > 10 * mse_values.min()) and mse_values.min() > 0
    
    if use_log:
        bars2 = axes[1].bar(range(len(best_df_sorted_mse)), np.log10(mse_values), 
                           color=plt.cm.Set3(np.arange(len(best_df_sorted_mse)) + 5))
        axes[1].set_ylabel('log10(MSE)', fontsize=14)
        ylabel_suffix = ' (escala log)'
    else:
        bars2 = axes[1].bar(range(len(best_df_sorted_mse)), mse_values, 
                           color=plt.cm.Set3(np.arange(len(best_df_sorted_mse)) + 5))
        axes[1].set_ylabel('MSE', fontsize=14)
        ylabel_suffix = ''
    
    axes[1].set_title(f'Comparação do MSE por Tipo de Modelo{ylabel_suffix}', 
                     fontsize=18, fontweight='bold', pad=20)
    axes[1].set_xlabel('Modelo', fontsize=14)
    axes[1].set_xticks(range(len(best_df_sorted_mse)))
    axes[1].set_xticklabels(best_df_sorted_mse['Modelo'], rotation=45, ha='right', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Adicionar valores nas barras
    for i, (bar, mse) in enumerate(zip(bars2, best_df_sorted_mse['MSE'])):
        if use_log:
            text_val = f'10^{np.log10(mse):.2f}'
        else:
            text_val = f'{mse:.2e}' if mse < 0.001 else f'{mse:.6f}'
        
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    text_val, ha='center', va='bottom', fontsize=10, fontweight='bold',
                    rotation=90 if len(text_val) > 8 else 0)
    
    plt.tight_layout()
    plt.savefig('comparacao_modelos.png', dpi=150)
    plt.show()
    
    return best_df


def plot_model_comparison_sem_redes_neurais(df, tecnicas):
    """
    Plota um gráfico separado comparando o R² de todos os modelos,
    EXCETO Redes Neurais.
    Salva em um arquivo PNG separado.
    """

    print("\n📊 Gerando gráfico de comparação de R² (SEM Redes Neurais)...")

    # Remover Redes Neurais da lista de técnicas
    tecnicas_filtradas = [t for t in tecnicas if t != 'Redes Neurais']

    best_by_model = []

    for modelo in tecnicas_filtradas:
        model_df = df[df['Modelo'] == modelo]

        if len(model_df) > 0:
            if not model_df['R²'].isna().all():
                best_idx = model_df['R²'].idxmax()
                best_by_model.append(model_df.loc[best_idx])

    if not best_by_model:
        print("❌ Nenhum dado válido encontrado para o gráfico sem Redes Neurais.")
        return

    best_df = pd.DataFrame(best_by_model)

    # Ordenar por R²
    best_df_sorted = best_df.sort_values('R²', ascending=False)

    plt.figure(figsize=(14, 8))

    bars = plt.bar(
        range(len(best_df_sorted)),
        best_df_sorted['R²']
    )

    plt.title(
        'Comparação do Melhor R² por Tipo de Modelo (SEM Redes Neurais)',
        fontsize=16,
        fontweight='bold'
    )
    plt.xlabel('Modelo', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)

    plt.xticks(
        range(len(best_df_sorted)),
        best_df_sorted['Modelo'],
        rotation=45,
        ha='right'
    )

    plt.grid(True, alpha=0.3, axis='y')

    # Valores nas barras
    for bar, r2 in zip(bars, best_df_sorted['R²']):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{r2:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()

    # Salvar em ARQUIVO DIFERENTE
    plt.savefig(
        'comparacao_modelos_R2_SEM_Redes_Neurais.png',
        dpi=150
    )

    plt.show()

    print("✅ Gráfico salvo em: comparacao_modelos_R2_SEM_Redes_Neurais.png")


def plot_top_10_models(df):
    """Plota os 10 melhores modelos independente do tipo"""

    # Remover linhas com R² inválido
    valid_df = df.dropna(subset=['R²']).copy()
    
    if len(valid_df) < 10:
        print(f"Apenas {len(valid_df)} modelos válidos encontrados. Plotando todos.")
        top_n = len(valid_df)
    else:
        top_n = 10
    
    # Ordenar pelos melhores R²
    top_models = valid_df.nlargest(top_n, 'R²')
    
    # Criar labels descritivos
    labels = []
    for _, row in top_models.iterrows():
        label = row['Modelo']
        if row['Reduzido'] == 'X':
            label += " (Reduzido)"
        if row['Clustering'] == 'X':
            label += " +Clust"
        labels.append(label)

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    
    # Gráfico 1: R² dos top modelos
    x_pos = np.arange(len(top_models))
    colors_r2 = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_models)))
    
    bars1 = axes[0].bar(x_pos, top_models['R²'], color=colors_r2, edgecolor='black', linewidth=1.5)
    axes[0].set_title(f'Top {top_n} Modelos - Melhores R² Scores', fontsize=20, fontweight='bold', pad=20)
    axes[0].set_xlabel('Modelo e Configuração', fontsize=16)
    axes[0].set_ylabel('R² Score', fontsize=16)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Adicionar linha de referência em 0
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Anotar valores
    for i, (bar, r2) in enumerate(zip(bars1, top_models['R²'])):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Gráfico 2: MSE dos top modelos
    colors_mse = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_models)))
    
    top_models = top_models.copy()
    top_models['MSE'] = top_models['MSE'].clip(lower=1e-8)

    # Verificar se precisa de escala log
    mse_values = top_models['MSE'].values
    mse_min, mse_max = mse_values.min(), mse_values.max()
    use_log_mse = mse_max > 10 * mse_min and mse_min > 0
    
    if use_log_mse:
        y_values = np.log10(mse_values)
        y_label = 'log10(MSE)'
    else:
        y_values = mse_values
        y_label = 'MSE'
    
    bars2 = axes[1].bar(x_pos, y_values, color=colors_mse, edgecolor='black', linewidth=1.5)
    axes[1].set_title(f'Top {top_n} Modelos - MSE Scores', fontsize=20, fontweight='bold', pad=20)
    axes[1].set_xlabel('Modelo e Configuração', fontsize=16)
    axes[1].set_ylabel(y_label, fontsize=16)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Anotar valores originais do MSE
    for i, (bar, mse) in enumerate(zip(bars2, top_models['MSE'])):
        if use_log_mse:
            text = f'10^{np.log10(mse):.2f}'
        else:
            text = f'{mse:.2e}' if mse < 0.001 else f'{mse:.6f}'
        
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    text, ha='center', va='bottom', fontsize=10, fontweight='bold',
                    rotation=90 if len(text) > 8 else 0)
    
    plt.tight_layout()
    plt.gca().set_xticklabels(labels, fontsize=10)
    plt.savefig(f'top_{top_n}_modelos.png', dpi=150)
    plt.show()
    
    # Mostrar tabela com os top modelos
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MODELOS - RESUMO")
    print('='*80)
    display_cols = ['Modelo', 'R²', 'MSE', 'Reduzido', 'Clustering']
    if 'Tempo Execução (s)' in top_models.columns:
        display_cols.append('Tempo Execução (s)')
    
    for col in ['max_depth', 'n_estimators', 'learning_rate', 'n_neighbors', 'hidden_layer_sizes']:
        if col in top_models.columns and not top_models[col].isna().all():
            display_cols.append(col)
    
    display_df = top_models[display_cols].copy()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    print(display_df.to_string(index=False))
    
    return top_models

def select_best_model(df):
    """
    Seleciona o melhor modelo com base em critérios robustos:
    1. R² mais alto (prioridade principal)
    2. MSE mais baixo (desempate)
    3. Maior estabilidade (menor variância se houver múltiplas execuções)
    4. Complexidade do modelo (preferência por modelos mais simples)
    """
    
    # Remover modelos com R² inválido
    valid_df = df.dropna(subset=['R²']).copy()
    
    if len(valid_df) == 0:
        print("Nenhum modelo válido encontrado!")
        return None
    
    # Critério 1: R² mais alto
    max_r2 = valid_df['R²'].max()
    candidates = valid_df[valid_df['R²'] == max_r2]
    
    if len(candidates) == 1:
        best_model = candidates.iloc[0]
        criterion = "Maior R²"
    else:
        # Critério 2: Menor MSE entre os com mesmo R²
        min_mse = candidates['MSE'].min()
        candidates = candidates[candidates['MSE'] == min_mse]
        
        if len(candidates) == 1:
            best_model = candidates.iloc[0]
            criterion = "Maior R² + Menor MSE"
        else:
            # Critério 3: Preferir modelos mais simples (menos parâmetros)
            # Heurística de complexidade: definir pesos para cada tipo de modelo
            complexity_scores = {
                'Regressão Linear': 1,
                'Regressão Linear Bayesiana': 2,
                'Árvore de Decisão': 3,
                'K-NN': 4,
                'Random Forest': 5,
                'XGBoost': 6,
                'Redes Neurais': 7
            }
            
            candidates['Complexity'] = candidates['Modelo'].map(complexity_scores)
            best_model = candidates.loc[candidates['Complexity'].idxmin()]
            criterion = "Maior R² + Menor MSE + Menor Complexidade"
    
    # Calcular score composto para ranking
    valid_df = valid_df.copy()
    
    # Normalizar R² (0 a 1)
    r2_min, r2_max = valid_df['R²'].min(), valid_df['R²'].max()
    if r2_max > r2_min:
        valid_df['R²_norm'] = (valid_df['R²'] - r2_min) / (r2_max - r2_min)
    else:
        valid_df['R²_norm'] = 1.0
    
    # Normalizar MSE invertido (0 a 1, onde 1 é melhor)
    mse_min, mse_max = valid_df['MSE'].min(), valid_df['MSE'].max()
    if mse_max > mse_min:
        valid_df['MSE_norm'] = 1 - ((valid_df['MSE'] - mse_min) / (mse_max - mse_min))
    else:
        valid_df['MSE_norm'] = 1.0
    
    # Score composto (60% R², 40% MSE)
    valid_df['Composite_Score'] = 0.6 * valid_df['R²_norm'] + 0.4 * valid_df['MSE_norm']
    
    # Verificar se o melhor modelo pelo critério é também o melhor pelo score composto
    composite_best = valid_df.loc[valid_df['Composite_Score'].idxmax()]
    
    print("\n" + "="*80)
    print("SELEÇÃO DO MELHOR MODELO - ANÁLISE")
    print("="*80)
    
    print(f"\nCritério de seleção usado: {criterion}")
    print(f"\nModelo Selecionado:")
    print(f"  Tipo: {best_model['Modelo']}")
    print(f"  R²: {best_model['R²']:.6f}")
    print(f"  MSE: {best_model['MSE']:.6e}")
    
    if best_model['Reduzido'] == 'X':
        print(f"  Dataset: Reduzido")
    else:
        print(f"  Dataset: Completo")
    
    if best_model['Clustering'] == 'X':
        print(f"  Clustering: Sim")
    else:
        print(f"  Clustering: Não")
    
    # Mostrar hiperparâmetros se disponíveis
    param_cols = [col for col in df.columns if col not in 
                 ['Reduzido', 'Clustering', 'Modelo', 'R²', 'MSE', 'Config_ID', 
                  'Regressão Linear', 'Árvore de Decisão', 'Random Forest', 
                  'XGBoost', 'K-NN', 'Regressão Linear Bayesiana', 'Redes Neurais',
                  '3-Fold', '4-Fold', '5-Fold', '6-Fold', '7-Fold']]
    
    print(f"\nHiperparâmetros do modelo selecionado:")
    for param in param_cols:
        if param in best_model.index and pd.notna(best_model[param]) and best_model[param] not in ['', 'N/A']:
            print(f"  {param}: {best_model[param]}")
    
    # Comparação com score composto
    print(f"\nVerificação por Score Composto (60% R², 40% MSE):")
    print(f"  Score do modelo selecionado: {valid_df.loc[best_model.name, 'Composite_Score']:.4f}")
    print(f"  Melhor score composto: {composite_best['Composite_Score']:.4f}")
    
    if best_model.name != composite_best.name:
        print(f"  Nota: O modelo com melhor score composto é diferente:")
        print(f"    Modelo: {composite_best['Modelo']} (Score: {composite_best['Composite_Score']:.4f})")
    
    return best_model

def load_original_data():
    """Carrega os dataframes originais usados no experimento"""
    print("\nCarregando dataframes originais para feature importance...")
    try:
        df_merged = pd.read_csv('df_merged.csv')
        df_merged_reduzido = pd.read_csv('df_merged_reduzido.csv')
        df_clusters = pd.read_csv('df_clusters.csv')
        df_clusters_reduzido = pd.read_csv('df_clusters_reduzido.csv')
        
        print("✅ Dataframes originais carregados com sucesso!")
        return {
            'df_merged': df_merged,
            'df_merged_reduzido': df_merged_reduzido,
            'df_clusters': df_clusters,
            'df_clusters_reduzido': df_clusters_reduzido
        }
    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar arquivos de dados originais: {e}")
        print("Certifique-se de que os arquivos estão no diretório atual:")
        print("  - df_merged.csv")
        print("  - df_merged_reduzido.csv")
        print("  - df_clusters.csv")
        print("  - df_clusters_reduzido.csv")
        return None

def selecionar_dataframe(reduzido, clustering, dataframes):
    """Seleciona o dataframe correto baseado nas configurações"""
    if clustering == 'X' and reduzido == 'X':
        return dataframes['df_clusters_reduzido']
    elif clustering == 'X' and reduzido != 'X':
        return dataframes['df_clusters']
    elif clustering != 'X' and reduzido == 'X':
        return dataframes['df_merged_reduzido']
    else:
        return dataframes['df_merged']

def plot_feature_importance_real(best_model_info):
    """
    Treina o melhor modelo com os hiperparâmetros explicitados no CSV
    e plota as feature importances reais.
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE REAL - TREINANDO MODELO COM HIPERPARÂMETROS")
    print("="*80)
    
    # 1. Carregar dados originais
    dataframes = load_original_data()
    if dataframes is None:
        print("❌ Não foi possível carregar os dados originais.")
        print("   A feature importance real não pode ser calculada.")
        return
    
    # 2. Selecionar dataframe correto
    reduzido = best_model_info['Reduzido']
    clustering = best_model_info['Clustering']
    df = selecionar_dataframe(reduzido, clustering, dataframes)
    
    # 3. Preparar dados
    print(f"\n📊 Preparando dados...")
    print(f"   Dataset selecionado: {'Reduzido' if reduzido == 'X' else 'Completo'}, "
          f"{'Com Clustering' if clustering == 'X' else 'Sem Clustering'}")
    
    X = df.drop(columns=['Taxa a cada 100 mil hab.', 'id_municipio'])
    y = df['Taxa a cada 100 mil hab.']
    
    # ============================
    # DEBUG: COLUNAS DE ENTRADA DO MODELO (ORDENADAS)
    # ============================
    feature_list = sorted(X.columns.tolist())

    print("\n" + "="*80)
    print("COLUNAS DO DATAFRAME DE ENTRADA DO MODELO (ORDENADAS ALFABETICAMENTE)")
    print("="*80)
    print(f"Total de features: {len(feature_list)}\n")

    for i, col in enumerate(feature_list, start=1):
        print(f"{i:3d}. {col}")

    print("="*80 + "\n")


    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    print(f"   Total de features: {n_features}")
    
    # 4. Criar modelo com hiperparâmetros do CSV
    modelo_tipo = best_model_info['Modelo']
    print(f"\n🔧 Configurando modelo: {modelo_tipo}")
    
    # Extrair hiperparâmetros do CSV
    param_dict = {}
    
    # Mapeamento de colunas de hiperparâmetros (baseado no seu CSV)
    param_columns = {
        'max_depth': 'max_depth',
        'min_samples_split': 'min_samples_split',
        'min_samples_leaf': 'min_samples_leaf',
        'max_features': 'max_features',
        'n_estimators': 'n_estimators',
        'learning_rate': 'learning_rate',
        'subsample': 'subsample',
        'colsample_bytree': 'colsample_bytree',
        'n_neighbors': 'n_neighbors',
        'weights': 'weights',
        'metric': 'metric',
        'alpha_1': 'alpha_1',
        'alpha_2': 'alpha_2',
        'lambda_1': 'lambda_1',
        'lambda_2': 'lambda_2',
        'hidden_layer_sizes': 'hidden_layer_sizes',
        'activation': 'activation',
        'alpha': 'alpha',
        'learning_rate_init': 'learning_rate_init'
    }
    
    for param_name, csv_col in param_columns.items():
        if csv_col in best_model_info.index and pd.notna(best_model_info[csv_col]) and best_model_info[csv_col] not in ['', 'N/A']:
            value = best_model_info[csv_col]
            
            # Conversão robusta de tipos
            if pd.isna(value):
                continue

            # Se for string
            if isinstance(value, str):
                v = value.strip().lower()

                if v == 'none':
                    value = None

                elif value.startswith('(') and value.endswith(')'):
                    try:
                        value = tuple(int(x.strip()) for x in value[1:-1].split(','))
                    except:
                        pass

                else:
                    try:
                        if '.' in value:
                            value = float(value)
                            if value.is_integer():
                                value = int(value)
                        else:
                            value = int(value)
                    except:
                        pass

            # Se for numpy float (caso do seu erro)
            elif isinstance(value, (np.floating, float)):
                if value.is_integer():
                    value = int(value)

            # Se for numpy int
            elif isinstance(value, (np.integer, int)):
                value = int(value)

            param_dict[param_name] = value
            print(f"   {param_name}: {value} ({type(value).__name__})")

            
            param_dict[param_name] = value
            print(f"   {param_name}: {value}")
    
    # 5. Instanciar e configurar o modelo
    print(f"\n🏗️  Instanciando modelo...")
    
    if modelo_tipo == 'Regressão Linear':
        model = LinearRegression(**param_dict)
        
    elif modelo_tipo == 'Árvore de Decisão':
        model = DecisionTreeRegressor(random_state=42, **param_dict)
        
    elif modelo_tipo == 'Random Forest':
        model = RandomForestRegressor(random_state=42, n_jobs=1, **param_dict)
        
    elif modelo_tipo == 'XGBoost':
        model = xgb.XGBRegressor(random_state=42, n_jobs=1, **param_dict)
        
    elif modelo_tipo == 'K-NN':
        model = KNeighborsRegressor(n_jobs=1, **param_dict)
        
    elif modelo_tipo == 'Regressão Linear Bayesiana':
        model = BayesianRidge(**param_dict)
        
    elif modelo_tipo == 'Redes Neurais':
        model = MLPRegressor(random_state=42, max_iter=1000, **param_dict)
        
    else:
        print(f"❌ Tipo de modelo não suportado: {modelo_tipo}")
        return
    
    # 6. Treinar modelo
    print(f"\n🚀 Treinando modelo...")
    model.fit(X, y)
    
    # 7. Extrair feature importances
    print(f"\n📈 Extraindo feature importances...")
    
    importance = None
    
    # Diferentes métodos para diferentes tipos de modelo
    if modelo_tipo in ['Regressão Linear', 'Regressão Linear Bayesiana']:
        # Modelos lineares: usar coeficientes absolutos
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            print("   Usando coeficientes absolutos (modelo linear)")
    
    elif modelo_tipo in ['Árvore de Decisão', 'Random Forest']:
        # Modelos baseados em árvore: usar feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            print("   Usando feature_importances_ (modelo baseado em árvore)")
    
    elif modelo_tipo == 'XGBoost':
        # XGBoost: usar feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            print("   Usando feature_importances_ (XGBoost)")
    
    elif modelo_tipo == 'K-NN':
        # K-NN não tem feature importance nativa
        # Usar permutation importance como alternativa
        print("   Calculando permutation importance (K-NN não tem feature importance nativa)...")
        perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance = perm_result.importances_mean
        print("   Permutation importance calculada")
    
    elif modelo_tipo == 'Redes Neurais':
        # Redes Neurais: usar permutation importance
        print("   Calculando permutation importance para Redes Neurais...")
        perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=1)
        importance = perm_result.importances_mean
        print("   Permutation importance calculada")
    
    if importance is None:
        print(f"❌ Não foi possível extrair feature importance para {modelo_tipo}")
        return
    
    # 8. Ordenar importâncias
    sorted_idx = np.argsort(importance)[::-1]  # Ordem decrescente
    sorted_importance = importance[sorted_idx]
    sorted_feature_names = [feature_names[i] for i in sorted_idx]
    
    # 9. Plotar gráficos
    print(f"\n🎨 Criando visualizações...")
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Top 30 features
    ax1 = plt.subplot(3, 3, 1)
    top_n = min(30, n_features)
    colors1 = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    bars1 = ax1.barh(range(top_n), sorted_importance[:top_n][::-1], color=colors1, edgecolor='black')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(sorted_feature_names[:top_n][::-1], fontsize=8)
    ax1.set_xlabel('Importância', fontsize=12)
    ax1.set_title(f'TOP 30 FEATURES MAIS IMPORTANTES', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars1, sorted_importance[:top_n][::-1])):
        ax1.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=7)
    
    # 2. Top 20 features
    ax2 = plt.subplot(3, 3, 2)
    top_n = min(20, n_features)
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, top_n))
    bars2 = ax2.barh(range(top_n), sorted_importance[:top_n][::-1], color=colors2, edgecolor='black')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(sorted_feature_names[:top_n][::-1], fontsize=9)
    ax2.set_xlabel('Importância', fontsize=12)
    ax2.set_title(f'TOP 20 FEATURES MAIS IMPORTANTES', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Top 10 features
    ax3 = plt.subplot(3, 3, 3)
    top_n = min(10, n_features)
    colors3 = plt.cm.summer(np.linspace(0.2, 0.8, top_n))
    bars3 = ax3.barh(range(top_n), sorted_importance[:top_n][::-1], color=colors3, edgecolor='black')
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels(sorted_feature_names[:top_n][::-1], fontsize=10)
    ax3.set_xlabel('Importância', fontsize=12)
    ax3.set_title(f'TOP 10 FEATURES MAIS IMPORTANTES', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Bottom 50 features (menos importantes)
    ax4 = plt.subplot(3, 3, 4)
    bottom_n = min(50, n_features)
    bottom_importance = sorted_importance[-bottom_n:][::-1]
    bottom_features = sorted_feature_names[-bottom_n:][::-1]
    
    colors4 = plt.cm.autumn(np.linspace(0.2, 0.8, bottom_n))
    bars4 = ax4.barh(range(bottom_n), bottom_importance, color=colors4, edgecolor='black')
    
    # Mostrar apenas alguns labels para evitar sobreposição
    tick_step = max(1, bottom_n // 15)
    ax4.set_yticks(range(0, bottom_n, tick_step))
    ax4.set_yticklabels(bottom_features[::tick_step], fontsize=8, rotation=0)
    ax4.set_xlabel('Importância', fontsize=12)
    ax4.set_title(f'50 FEATURES MENOS IMPORTANTES', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Distribuição cumulativa da importância
    ax5 = plt.subplot(3, 3, (5, 6))
    cumulative_importance = np.cumsum(sorted_importance) / np.sum(sorted_importance)
    
    ax5.plot(range(1, n_features + 1), cumulative_importance * 100, 
             linewidth=3, color='darkblue', marker='o', markersize=4, markevery=20)
    ax5.fill_between(range(1, n_features + 1), 0, cumulative_importance * 100, alpha=0.2, color='blue')
    
    # Adicionar linhas de referência
    for pct in [50, 80, 90, 95]:
        idx = np.where(cumulative_importance >= pct/100)[0]
        if len(idx) > 0:
            n_features_pct = idx[0] + 1
            ax5.axvline(x=n_features_pct, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax5.axhline(y=pct, color='gray', linestyle=':', alpha=0.5, linewidth=2)
            ax5.text(n_features_pct, pct + 2, f'{n_features_pct} features\n({pct}%)', 
                    fontsize=10, ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax5.set_xlabel('Número de Features', fontsize=14)
    ax5.set_ylabel('Importância Acumulada (%)', fontsize=14)
    ax5.set_title('DISTRIBUIÇÃO CUMULATIVA DA IMPORTÂNCIA DAS FEATURES', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 105)
    ax5.set_xlim(0, n_features)
    
    # 6. Top 15 features com maiores importâncias (gráfico de pizza)
    ax6 = plt.subplot(3, 3, 7)
    top_n_pie = min(15, n_features)
    top_importance = sorted_importance[:top_n_pie]
    top_features_pie = sorted_feature_names[:top_n_pie]
    
    # Calcular porcentagens
    total_importance = np.sum(sorted_importance)
    percentages = (top_importance / total_importance) * 100
    
    # Criar gráfico de pizza
    wedges, texts, autotexts = ax6.pie(percentages, labels=top_features_pie, autopct='%1.1f%%',
                                        startangle=90, counterclock=False, 
                                        textprops={'fontsize': 8})
    
    # Melhorar legibilidade
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax6.set_title(f'TOP 15 FEATURES - DISTRIBUIÇÃO DA IMPORTÂNCIA', fontsize=12, fontweight='bold')
    
    # 7. Importância por categoria (se houver categorias nas features)
    ax7 = plt.subplot(3, 3, (8, 9))


# OBS: esta seção de código cria um gráfico adicional, mas estraga a disposição dos subplots. 
# Portanto, ela foi usada uma vez para gerar o gráfico separado.


    # # ============================================================
    # # NOVO GRÁFICO: IMPORTÂNCIA POR CATEGORIA (TAXA REFINADA)
    # # ============================================================

    # # Substrings que definem ESCOLARIDADE
    # substrings_escolaridade = [
    #     'escolaridade', 'estudo', 'analfabetismo', 'atraso', 'freq',
    #     'fundamental', 'medio', 'escola', 'escolar', 'idhm_e'
    # ]

    # def eh_escolaridade(nome_feature):
    #     nome = nome_feature.lower()
    #     return any(sub in nome for sub in substrings_escolaridade)

    # # Novo agrupamento refinado
    # feature_categories_refinado = {}

    # for feature in sorted_feature_names:
    #     nome = feature.lower()

    #     if 'taxa' in nome:
    #         # Subdividir a antiga categoria "taxa"
    #         if eh_escolaridade(nome):
    #             categoria = 'taxa_escolaridade'
    #         else:
    #             categoria = 'taxa_demografica'
    #     else:
    #         # Manter a lógica antiga de prefixo
    #         if '_' in feature:
    #             categoria = feature.split('_')[0]
    #         else:
    #             categoria = 'Outros'

    #     idx = sorted_feature_names.index(feature)

    #     if categoria not in feature_categories_refinado:
    #         feature_categories_refinado[categoria] = []

    #     feature_categories_refinado[categoria].append(sorted_importance[idx])

    # # Somar importância por categoria refinada
    # cat_names_ref = []
    # cat_importance_ref = []

    # for categoria, importancias in feature_categories_refinado.items():
    #     cat_names_ref.append(categoria)
    #     cat_importance_ref.append(np.sum(importancias))

    # # Ordenar por importância
    # cat_sorted_idx_ref = np.argsort(cat_importance_ref)[::-1]
    # cat_sorted_names_ref = [cat_names_ref[i] for i in cat_sorted_idx_ref]
    # cat_sorted_importance_ref = [cat_importance_ref[i] for i in cat_sorted_idx_ref]

    # # Criar nova figura APENAS para este gráfico
    # plt.figure(figsize=(18, 8))

    # bars_ref = plt.bar(
    #     range(len(cat_sorted_names_ref)),
    #     cat_sorted_importance_ref
    # )

    # plt.xlabel('Categoria (Refinada)', fontsize=14)
    # plt.ylabel('Importância Total', fontsize=14)
    # plt.title('IMPORTÂNCIA POR CATEGORIA (TAXA REFINADA)', fontsize=16, fontweight='bold')

    # plt.xticks(
    #     range(len(cat_sorted_names_ref)),
    #     cat_sorted_names_ref,
    #     rotation=45,
    #     ha='right',
    #     fontsize=11
    # )

    # plt.grid(True, alpha=0.3, axis='y')

    # # Adicionar valores nas barras
    # for bar, val in zip(bars_ref, cat_sorted_importance_ref):
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height(),
    #         f'{val:.3f}',
    #         ha='center',
    #         va='bottom',
    #         fontsize=10
    #     )

    # plt.tight_layout()

    # # Salvar como NOVO arquivo (não sobrescreve o anterior)
    # plt.savefig(
    #     f'feature_importance_categoria_REFINADA_{modelo_tipo.replace(" ", "_")}.png',
    #     dpi=150
    # )

    # plt.show()




    
    # Agrupar features por prefixos comuns (se existirem)
    feature_categories = {}
    for feature in sorted_feature_names:
        # Extrair categoria baseada em prefixos comuns
        if '_' in feature:
            category = feature.split('_')[0]
        else:
            category = 'Outros'
        
        idx = sorted_feature_names.index(feature)
        if category not in feature_categories:
            feature_categories[category] = []
        feature_categories[category].append(sorted_importance[idx])
    
    # Calcular importância total por categoria
    category_names = []
    category_importance = []
    for category, importances in feature_categories.items():
        category_names.append(category)
        category_importance.append(np.sum(importances))
    
    # Ordenar categorias por importância
    cat_sorted_idx = np.argsort(category_importance)[::-1]
    cat_sorted_names = [category_names[i] for i in cat_sorted_idx]
    cat_sorted_importance = [category_importance[i] for i in cat_sorted_idx]
    
    # Plotar importância por categoria
    bars_cat = ax7.bar(range(len(cat_sorted_names)), cat_sorted_importance, 
                       color=plt.cm.tab20c(np.arange(len(cat_sorted_names))))
    
    ax7.set_xlabel('Categoria', fontsize=14)
    ax7.set_ylabel('Importância Total', fontsize=14)
    ax7.set_title('IMPORTÂNCIA POR CATEGORIA DE FEATURES', fontsize=14, fontweight='bold')
    ax7.set_xticks(range(len(cat_sorted_names)))
    ax7.set_xticklabels(cat_sorted_names, rotation=45, ha='right', fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars_cat, cat_sorted_importance)):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'ANÁLISE COMPLETA DE FEATURE IMPORTANCE - {modelo_tipo.upper()}\n'
                 f'Dataset: {"Reduzido" if reduzido == "X" else "Completo"} | '
                 f'Clustering: {"Sim" if clustering == "X" else "Não"}', 
                 fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_REAL_{modelo_tipo.replace(" ", "_")}.png', 
                dpi=150, facecolor='white')
    plt.show()
    
    # 10. Estatísticas detalhadas
    print(f"\n{'='*80}")
    print("ESTATÍSTICAS DETALHADAS DAS FEATURE IMPORTANCES")
    print('='*80)
    
    print(f"\n📊 Estatísticas gerais:")
    print(f"   Total de features: {n_features}")
    print(f"   Importância total: {np.sum(sorted_importance):.6f}")
    print(f"   Importância média: {np.mean(sorted_importance):.6f}")
    print(f"   Importância mediana: {np.median(sorted_importance):.6f}")
    print(f"   Desvio padrão: {np.std(sorted_importance):.6f}")
    print(f"   Coeficiente de variação: {np.std(sorted_importance)/np.mean(sorted_importance):.3f}")
    
    print(f"\n📈 Features necessárias para alcançar:")
    for pct in [25, 50, 75, 80, 90, 95, 99]:
        idx = np.where(cumulative_importance >= pct/100)[0]
        if len(idx) > 0:
            n_needed = idx[0] + 1
            print(f"   {pct}% da importância: {n_needed} features ({n_needed/n_features*100:.1f}% do total)")
    
    print(f"\n🏆 TOP 10 FEATURES (mais importantes):")
    for i in range(min(10, n_features)):
        print(f"   {i+1:2d}. {sorted_feature_names[i]:<40} : {sorted_importance[i]:.6f} "
              f"({sorted_importance[i]/np.sum(sorted_importance)*100:.2f}%)")
    
    print(f"\n📉 BOTTOM 10 FEATURES (menos importantes):")
    for i in range(min(10, n_features)):
        idx = n_features - i - 1
        print(f"   {i+1:2d}. {sorted_feature_names[idx]:<40} : {sorted_importance[idx]:.6f} "
              f"({sorted_importance[idx]/np.sum(sorted_importance)*100:.4f}%)")
    
    # Salvar lista completa em arquivo
    with open(f'feature_importance_list_{modelo_tipo.replace(" ", "_")}.txt', 'w') as f:
        f.write(f"FEATURE IMPORTANCE - {modelo_tipo}\n")
        f.write(f"Dataset: {'Reduzido' if reduzido == 'X' else 'Completo'} | "
                f"Clustering: {'Sim' if clustering == 'X' else 'Não'}\n")
        f.write("="*100 + "\n")
        f.write(f"{'Posição':<8} {'Feature':<50} {'Importância':<15} {'% Total':<10} {'% Acumulado':<12}\n")
        f.write("-"*100 + "\n")
        
        cumulative = 0
        for i in range(n_features):
            cumulative += sorted_importance[i]
            f.write(f"{i+1:<8} {sorted_feature_names[i]:<50} "
                   f"{sorted_importance[i]:<15.6f} "
                   f"{sorted_importance[i]/np.sum(sorted_importance)*100:<10.4f} "
                   f"{cumulative/np.sum(sorted_importance)*100:<12.4f}\n")
    
    print(f"\n💾 Lista completa salva em: 'feature_importance_list_{modelo_tipo.replace(' ', '_')}.txt'")
    print(f"📊 Gráficos salvos como: 'feature_importance_REAL_{modelo_tipo.replace(' ', '_')}.png'")

def main():
    """Função principal"""
    
    # Solicitar arquivo ao usuário
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("Digite o caminho para o arquivo CSV de resultados: ").strip()
    
    try:
        # Carregar e preparar dados
        print(f"\nCarregando dados de: {filepath}")
        df, tecnicas = load_and_prepare_data(filepath)
        
        print(f"\nDados carregados com sucesso!")
        print(f"Total de registros: {len(df)}")
        print(f"Modelos encontrados: {df['Modelo'].unique()}")
        print(f"Colunas disponíveis: {list(df.columns)}")
        
        # Análise 1: Comparação entre modelos
        print("\n" + "="*80)
        print("ANÁLISE 1: COMPARAÇÃO ENTRE DIFERENTES TIPOS DE MODELOS")
        print("="*80)
        
        best_by_model_df = plot_model_comparison(df, tecnicas)
        plot_model_comparison_sem_redes_neurais(df, tecnicas)


        # Análise 2: Top 10 modelos
        print("\n" + "="*80)
        print("ANÁLISE 2: TOP 10 MODELOS (MELHORES R²)")
        print("="*80)
        top_10_df = plot_top_10_models(df)
        
        # Análise 3: Seleção do melhor modelo
        print("\n" + "="*80)
        print("ANÁLISE 3: SELEÇÃO DO MELHOR MODELO")
        print("="*80)
        best_model = select_best_model(df)
        
        if best_model is not None:
            # Análise 4: Feature Importance (simulação)
            print("\n" + "="*80)
            print("ANÁLISE 4: FEATURE IMPORTANCE DO MELHOR MODELO")
            print("="*80)
            plot_feature_importance_real(best_model)
            
            # Salvar informações do melhor modelo
            print("\n" + "="*80)
            print("RESUMO FINAL")
            print("="*80)
            print(f"\n🎯 MELHOR MODELO SELECIONADO: {best_model['Modelo']}")
            print(f"   R²: {best_model['R²']:.6f}")
            print(f"   MSE: {best_model['MSE']:.6e}")
            
            # Criar um pequeno relatório
            with open('melhor_modelo_relatorio.txt', 'w') as f:
                f.write("="*60 + "\n")
                f.write("RELATÓRIO DO MELHOR MODELO\n")
                f.write("="*60 + "\n\n")
                f.write(f"Modelo: {best_model['Modelo']}\n")
                f.write(f"R² Score: {best_model['R²']:.6f}\n")
                f.write(f"MSE: {best_model['MSE']:.6e}\n")
                f.write(f"Dataset: {'Reduzido' if best_model['Reduzido'] == 'X' else 'Completo'}\n")
                f.write(f"Clustering: {'Sim' if best_model['Clustering'] == 'X' else 'Não'}\n\n")
                f.write("Hiperparâmetros:\n")
                
                # Escrever hiperparâmetros
                param_cols = [col for col in df.columns if col not in 
                            ['Reduzido', 'Clustering', 'Modelo', 'R²', 'MSE', 'Config_ID',
                             'Regressão Linear', 'Árvore de Decisão', 'Random Forest', 
                             'XGBoost', 'K-NN', 'Regressão Linear Bayesiana', 'Redes Neurais',
                             '3-Fold', '4-Fold', '5-Fold', '6-Fold', '7-Fold']]
                
                for param in param_cols:
                    if param in best_model.index and pd.notna(best_model[param]) and best_model[param] not in ['', 'N/A']:
                        f.write(f"  {param}: {best_model[param]}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("RECOMENDAÇÕES:\n")
                f.write("="*60 + "\n")
                f.write("1. Este modelo apresentou o melhor equilíbrio entre R² e MSE.\n")
                f.write("2. Considere validar em um conjunto de testes independente.\n")
                f.write("3. Para feature importance real, retreine o modelo com os dados originais.\n")
                
            print(f"\n📄 Relatório salvo em: 'melhor_modelo_relatorio.txt'")
        
        print("\n✅ Análise concluída com sucesso!")
        print("📊 Gráficos salvos como PNG na pasta atual.")
        
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo '{filepath}' não encontrado!")
    except pd.errors.EmptyDataError:
        print("❌ Erro: O arquivo CSV está vazio!")
    except Exception as e:
        print(f"❌ Erro inesperado: {str(e)}")

if __name__ == "__main__":
    main()