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

# Tenta forçar um backend interativo
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
# Aumenta o tamanho do gráfico
plt.figure(figsize=(18, 8)) 

# Define o eixo X com os índices temporais
x_values = df['time_idx'].values
# Rótulos para TODAS as semanas
x_labels = [f"Sem.{s}\n{a}" for s, a in zip(df['semana_num'], df['ano'])]


# 1.1. Casos Reais (para contexto visual)
plt.plot(x_values, df['real_casos'], label='Casos Reais (Série Cinza)', color='gray', linestyle='-', alpha=0.5)

# 1.2. Plotagem da Classificação Real e Predita
colors = {0: 'green', 1: 'red'} # 0=Baixo Risco (verde), 1=Alto Risco (vermelho)

# Define as posições Y fixas para as classes 
Y_REAL_ALTO = 30.0
Y_REAL_BAIXO = 10.0
Y_PRED_ALTO = 70.0
Y_PRED_BAIXO = 50.0

# Classificação Real: Pista Inferior
y_offset_real = df['real_class'].replace({0: Y_REAL_BAIXO, 1: Y_REAL_ALTO}) 
plt.scatter(x_values, y_offset_real, 
            marker='o', s=80, 
            c=df['real_class'].map(colors), 
            label='Classe Real (o)', alpha=0.9)

# Classificação Predita: Pista Superior
y_offset_pred = df['pred_class'].replace({0: Y_PRED_BAIXO, 1: Y_PRED_ALTO}) 
plt.scatter(x_values, y_offset_pred, 
            marker='x', s=150, 
            c=df['pred_class'].map(colors), 
            linewidths=3, label='Classe Predita (x)')

# Adiciona linhas de referência pontilhadas para as áreas de classificação
plt.axhline(y=Y_REAL_ALTO, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.axhline(y=Y_REAL_BAIXO, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.axhline(y=Y_PRED_ALTO, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
plt.axhline(y=Y_PRED_BAIXO, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)


# --- Rótulos do Eixo X: Todas as semanas com rotação de 90 graus ---
plt.xticks(x_values, x_labels, rotation=90, ha='center')
plt.tick_params(axis='x', which='major', pad=10) 


# --- Rótulos do Eixo Y: Apenas escala de Casos. Removemos os textos de classificação! ---

# Mantemos apenas os ticks numéricos para a série de casos (acima de Y_PRED_ALTO)
y_ticks_casos = [t for t in plt.yticks()[0] if t > Y_PRED_ALTO] 
plt.yticks(y_ticks_casos, [str(int(t)) for t in y_ticks_casos]) 

# Títulos e Layout
plt.title("Classificação de Risco de Dengue (Real vs. Predito) - Período de Teste", fontsize=14)
plt.xlabel("Semanas Epidemiológicas", fontsize=12)
plt.ylabel("Casos (Série Cinza)", fontsize=12)
plt.ylim(0, df['real_casos'].max() * 1.1) 

# Ajusta a legenda para ser mais informativa: adiciona o significado da cor (o que antes estava no texto embolado)
plt.legend(loc='upper right', handles=[
    plt.Line2D([0], [0], color='gray', linestyle='-', alpha=0.5, label='Casos Reais (Série Cinza)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', markersize=10, label='Alto Risco (Real)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markeredgecolor='black', markersize=10, label='Baixo Risco (Real)'),
    plt.Line2D([0], [0], marker='x', color='red', markeredgecolor='red', markersize=10, linewidth=3, label='Alto Risco (Predito)'),
    plt.Line2D([0], [0], marker='x', color='green', markeredgecolor='green', markersize=10, linewidth=3, label='Baixo Risco (Predito)')
])


plt.grid(axis='y', linestyle='--')
plt.tight_layout()

# Salva o gráfico de série temporal
plot_path_series = os.path.join("outputs", "classification_series_plot.png")
plt.savefig(plot_path_series, bbox_inches="tight")
plt.close()

# --- 2. Heatmap da Matriz de Confusão (Mantido) ---
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

plt.show() 

print(f"✅ Gráfico de Série Temporal gerado: {plot_path_series}")
print(f"✅ Heatmap da Matriz de Confusão gerado: {plot_path_cm}")