"""
desafio2_dengue_nn.py

Treina uma Rede Neural Artificial (ANN) para CLASSIFICAÇÃO de semanas de dengue
(Alto ou Baixo número de casos). Este script atende aos requisitos do Desafio 2.

Uso:
    # 1. Instale as dependências:
    # pip install -r requirements_nn.txt 
    
    # 2. Execute o treinamento e gere os arquivos:
    # python desafio2_dengue_nn.py --input dengue.csv --output outputs --seed 42
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Garante a reprodutibilidade, crucial para Redes Neurais
tf.random.set_seed(42)
np.random.seed(42)

# --- Funções de Pré-processamento e Criação de Alvo ---

def load_and_clean(path: str) -> pd.DataFrame:
    """
    CORRIGIDO: Carrega os dados, limpa, FILTRA as linhas de resumo e 
    cria o alvo de CLASSIFICAÇÃO (is_high_case).
    """
    df = pd.read_csv(path, encoding="latin1", sep=";", skiprows=4, na_values='-')
    df = df.rename(columns=lambda c: c.strip())
    semana_cols = [c for c in df.columns if c.lower().startswith('semana')]
    
    # 1. Pré-filtragem na coluna 'Ano notificação'
    # Converte para string, remove aspas e filtra apenas linhas que parecem conter um ano (4 dígitos)
    df['Ano notificação_str'] = df['Ano notificação'].astype(str).str.replace('"', '').str.strip()
    df_filtered = df[df['Ano notificação_str'].str.match(r'^\d{4}$')]

    # 2. Pivotar os dados para o formato longo (semana a semana)
    df_long = df_filtered[['Ano notificação_str'] + semana_cols].melt(
        id_vars=['Ano notificação_str'],
        value_vars=semana_cols,
        var_name='semana',
        value_name='casos'
    )
    
    # 3. Conversão segura para inteiro
    df_long = df_long.rename(columns={'Ano notificação_str': 'ano'})
    df_long['ano'] = df_long['ano'].astype(int) 
    
    df_long['semana_num'] = df_long['semana'].str.extract(r'(\d+)').astype(int)
    df_long['casos'] = pd.to_numeric(df_long['casos'], errors='coerce').fillna(0).astype(int)
    df_long = df_long.sort_values(['ano', 'semana_num']).reset_index(drop=True)

    # NOVO ALVO DE CLASSIFICAÇÃO: Alto risco (1) ou Baixo Risco (0)
    # Define o limiar como a mediana dos casos (excluindo semanas com 0 caso para um limiar mais significativo)
    median_cases = df_long[df_long['casos'] > 0]['casos'].median()
    df_long['is_high_case'] = (df_long['casos'] > median_cases).astype(int)
    
    first_year = sorted(df_long['ano'].unique())[0]
    df_long['time_idx'] = (df_long['ano'] - first_year) * 52 + (df_long['semana_num'] - 1)
    df_out = df_long[['ano', 'semana_num', 'casos', 'time_idx', 'is_high_case']].copy()
    
    print(f"  > Mediana de casos para o corte de classificação: {median_cases:.2f}")
    print(f"  > Distribuição de Classes (0=Baixo, 1=Alto): {df_out['is_high_case'].value_counts().to_dict()}")

    return df_out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de lag, média móvel e sazonais para a Rede Neural."""
    out = df.copy()
    for lag in [1, 2, 3, 52]:
        out[f'lag_{lag}'] = out['casos'].shift(lag)
    out['rolling_mean_4'] = out['casos'].shift(1).rolling(window=4, min_periods=1).mean()
    out['rolling_std_4'] = out['casos'].shift(1).rolling(window=4, min_periods=1).std().fillna(0)
    out['semana_sin'] = np.sin(2 * np.pi * out['semana_num'] / 52)
    out['semana_cos'] = np.cos(2 * np.pi * out['semana_num'] / 52)
    
    # Remove linhas com NaNs (ocorrências iniciais)
    required_cols = [c for c in out.columns if c.startswith('lag_') or c.startswith('rolling_')]
    out = out.dropna(subset=required_cols).reset_index(drop=True)
    return out


# --- Função de Treinamento e Avaliação da ANN ---

def train_and_evaluate_nn(df_features: pd.DataFrame, seed: int = 42, output_dir: str = 'outputs'):
    """Constrói, treina e avalia a Rede Neural, salvando o modelo e os resultados."""
    
    # 1. Preparação dos dados
    features_to_drop = ['casos', 'is_high_case', 'ano', 'time_idx']
    X = df_features.drop(columns=features_to_drop)
    y = df_features['is_high_case']

    # Divisão treino/teste (mantendo a ordem temporal)
    test_size_ratio = 0.20
    test_size = max(1, int(len(X) * test_size_ratio))
    # NOTA: O Random Forest do Desafio 1 usava a parte final para teste, mantemos essa lógica temporal.
    X_train_raw, X_test_raw = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # Padronização dos dados (crucial para o bom desempenho de ANNs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_nn.pkl'))

    # 2. Criação do modelo (2 camadas ocultas conforme requisito)
    input_dim = X_train_scaled.shape[1]
    model = Sequential([
        # 1ª Camada Oculta
        Dense(64, activation='relu', input_shape=(input_dim,)),
        # 2ª Camada Oculta
        Dense(32, activation='relu'), 
        # Camada de Saída (classificação binária)
        Dense(1, activation='sigmoid') 
    ])

    # 3. Compilação do modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # Perda binária
                  metrics=['accuracy'])

    # 4. Treinamento
    print(f"  > Treinando o modelo com {len(X_train_scaled)} amostras...")
    model.fit(X_train_scaled, y_train, 
              epochs=50, 
              batch_size=32, 
              validation_data=(X_test_scaled, y_test),
              verbose=0)

    # 5. Avaliação e Métricas
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int) # Limite de 0.5
    
    # Métricas
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Salvar o modelo e os resultados para plotting
    model_path = os.path.join(output_dir, 'ann_model.h5')
    model.save(model_path) 
    
    test_results = pd.DataFrame({
        'time_idx': df_features.loc[X_test_raw.index, 'time_idx'], 
        'ano': df_features.loc[X_test_raw.index, 'ano'],
        'semana_num': df_features.loc[X_test_raw.index, 'semana_num'],
        'real_casos': df_features.loc[X_test_raw.index, 'casos'],
        'real_class': y_test,
        'pred_class': y_pred.flatten()
    }).reset_index(drop=True)
    test_results_path = os.path.join(output_dir, 'classification_test_results.csv')
    test_results.to_csv(test_results_path, index=False)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'conf_matrix': conf_matrix.tolist()
    }

    return model_path, metrics, test_results_path


def main_nn(args):
    print("🔍 Desafio 2: Iniciando o projeto de Classificação de Dengue com ANN.")
    df_long = load_and_clean(args.input)

    print("⚙️  Gerando e preparando features...")
    df_features = create_features(df_long)
    
    print("🧠 Construindo e treinando a Rede Neural Artificial (ANN)...")
    model_path, metrics, results_path = train_and_evaluate_nn(df_features, seed=args.seed, output_dir=args.output)
    
    # Geração de resultados
    print("\n\n--- Resultados da CLASSIFICAÇÃO (ANN) ---")
    print(f"✅ Modelo treinado e salvo em: {model_path}")
    print(f"✅ Resultados do teste salvos para plotagem: {results_path}")
    print("\n📊 Métricas de Classificação no Conjunto de Teste:")
    print(f"  Acurácia: {metrics['accuracy']:.4f}")
    print(f"  Precisão: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print("\n  Matriz de Confusão:")
    print(f"    {metrics['conf_matrix']}")
    print("    (Linhas: Real [0=Baixo, 1=Alto], Colunas: Predito [0=Baixo, 1=Alto])")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina ANN para Classificação de Casos de Dengue')
    parser.add_argument('--input', type=str, default='dengue.csv', help='Caminho para o CSV (formato do SINAN)')
    parser.add_argument('--output', type=str, default='outputs', help='Diretório de saída')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reprodutibilidade')
    args = parser.parse_args()
    main_nn(args)