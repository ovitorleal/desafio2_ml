"""
plot_classification_results.py
Gera um gráfico e um heatmap que visualizam os resultados do modelo de classificação (ANN).
Requer o arquivo 'outputs/classification_test_results.csv' gerado pelo script desafio2_dengue_nn.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import numpy as np

# Tenta forçar um backend interativo (útil em alguns ambientes que não abrem a janela)
try:
    plt.switch_backend('TkAgg') 
except ImportError:
    pass

# Caminho do arquivo de resultados de teste
csv_path = os.path.join("outputs", "classification_test_results.csv")

# Verifica se o arquivo existe
if not os.path.exists(csv_path):
    print("❌ O arquivo 'outputs/classification_test_results.csv' não foi encontrado. Execute o desafio2_dengue_nn.py antes.")
    exit()

# Carrega os resultados
df = pd.read_csv(csv_path)

# --- 1. Gráfico de Série Temporal (Casos Reais e Classificação) ---
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))

# Define o eixo X com os índices temporais
x_values = df['time_idx'].values # Usar .values para garantir que seja um numpy array
x_labels = [f"Sem.{s}\n{a}" for s, a in zip(df['semana_num'], df['ano'])]


# 1.1. Casos Reais (para contexto visual)
plt.plot(x_values, df['real_casos'], label='Casos Reais (Base)', color='gray', linestyle='-', alpha=0.5)

# 1.2. Plotagem da Classificação Real e Predita
colors = ['green', 'red'] 

# Classificação Real: Ajusta a posição Y para a "pista" inferior
y_offset_real = df['real_class'].replace({0: 10, 1: 30}) 
plt.scatter(x_values, y_offset_real, 
            marker='o', s=60, 
            c=[colors[c] for c in df['real_class']], 
            label='Classe Real (o)', alpha=0.7)

# Classificação Predita: Ajusta a posição Y para a "pista" superior
y_offset_pred = df['pred_class'].replace({0: 50, 1: 70}) 
plt.scatter(x_values, y_offset_pred, 
            marker='x', s=100, 
            c=[colors[c] for c in df['pred_class']], 
            linewidths=2, label='Classe Predita (x)')

# Rótulos e Títulos
plt.title("Classificação de Risco de Dengue (Real vs. Predito) - Período de Teste", fontsize=14)
plt.xlabel("Semanas Epidemiológicas", fontsize=12)
plt.ylabel("Casos (Série Cinza)", fontsize=12)

# Define os ticks e labels do eixo Y
plt.yticks([10, 30, 50, 70] + list(plt.yticks()[0]), 
           ['Baixo Risco (Real)', 'Alto Risco (Real)', 'Baixo Risco (Predito)', 'Alto Risco (Predito)'] + [''] * len(plt.yticks()[0]))
           
plt.xticks(x_values, x_labels, rotation=45, ha='right')

plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

# Salva o gráfico de série temporal
plot_path_series = os.path.join("outputs", "classification_series_plot.png")
plt.savefig(plot_path_series, bbox_inches="tight")
plt.close()

# --- 2. Heatmap da Matriz de Confusão ---
cm = confusion_matrix(df['real_class'], df['pred_class'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baixo (Predito)', 'Alto (Predito)'], 
            yticklabels=['Baixo (Real)', 'Alto (Real)'])
plt.title('Matriz de Confusão do Modelo ANN', fontsize=14)
plt.ylabel('Classe Real', fontsize=12)
plt.xlabel('Classe Predita', fontsize=12)
plt.tight_layout()

# Salva o gráfico de matriz de confusão
plot_path_cm = os.path.join("outputs", "confusion_matrix_heatmap.png")
plt.savefig(plot_path_cm, bbox_inches="tight")

# CORREÇÃO PARA ABRIR O GRÁFICO:
# Chamamos plt.show() no final do script para forçar a abertura das janelas.
# Isso garante que as janelas sejam exibidas após todos os gráficos serem criados e salvos.
plt.show()


print(f"✅ Gráfico de Série Temporal gerado: {plot_path_series}")
print(f"✅ Heatmap da Matriz de Confusão gerado: {plot_path_cm}")