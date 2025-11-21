# 🦠 Classificação de Risco de Dengue — Rede Neural Artificial (ANN)

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Rede Neural](https://img.shields.io/badge/Machine%20Learning-Keras/TensorFlow-red?logo=tensorflow)
![Objetivo](https://img.shields.io/badge/Objetivo-Classifica%C3%A7%C3%A3o-orange)
![Status](https://img.shields.io/badge/status-Em_desenvolvimento-yellow)
![Licença](https://img.shields.io/badge/Licença-Livre-lightgrey)

---

## 🎯 Objetivo do Projeto

Este repositório entrega o Desafio 2 do módulo de Machine Learning, aplicando uma Rede Neural Artificial (ANN) para classificação de risco de surto de dengue.

O modelo:

- Classifica cada semana epidemiológica como:
  - Alto Risco (1)
  - Baixo Risco (0)
- Define o limiar usando a mediana de casos.
- Utiliza uma Rede Neural Artificial com pelo menos 2 camadas ocultas.
- Avalia o desempenho utilizando a métrica de Acurácia.

---

## 💡 Contexto e Motivação

Na Vigilância Epidemiológica, a velocidade para emitir alertas é essencial.  
Em vez de prever a quantidade exata de casos (regressão), este projeto responde uma pergunta crucial:

**“A próxima semana terá alto risco de dengue?”**

Isso transforma a vigilância em um sistema de alerta rápido, permitindo ações de prevenção antes da explosão dos casos.

---

## 📁 Estrutura do Projeto

📂 dengue-ml/  
┣ desafio2_dengue_nn.py — Treina a ANN e gera classificação binária  
┣ plot_classification_results.py — Gera gráficos da classificação  
┣ requirements_nn.txt — Bibliotecas necessárias (TensorFlow/Keras)  
┣ dengue.csv — Base de dados original  
┗ 📂 outputs/  
  ┣ ann_model.h5 — Modelo treinado (Keras)  
  ┣ scaler_nn.pkl — Scaler para padronização  
  ┣ classification_test_results.csv — Resultados brutos do teste  
  ┣ classification_series_plot.png — Séries temporais: casos e classes  
  ┗ confusion_matrix_heatmap.png — Heatmap da Matriz de Confusão  

---

## ⚙️ Como Executar o Projeto

1️⃣ Instale as dependências  
pip install -r requirements_nn.txt

2️⃣ Treine o modelo e gere os resultados  
python desafio2_dengue_nn.py --input dengue.csv --output outputs --seed 42

3️⃣ Gere os gráficos de visualização  
python plot_classification_results.py

---

## 🧠 Arquitetura da Rede Neural (ANN)

A ANN foi construída com:

Camada    | Neurônios | Ativação  
--------- | ---------- | ---------  
Oculta 1  | 64         | ReLU  
Oculta 2  | 32         | ReLU  
Saída     | 1          | Sigmoid  

A saída é probabilística e convertida para classes 0 ou 1.

---

## 📊 Validação e Resultados

A principal métrica utilizada é a Acurácia.

O projeto gera:

- Matriz de Confusão — mostra TP, TN, FP e FN  
- Gráfico de Série Temporal comparando casos reais e classificação de risco  
- Arquivo CSV com os resultados do teste  

Essas visualizações ajudam a verificar:

- Taxa de acertos do modelo  
- Consistência da classificação ao longo do tempo  
- Desempenho real no processo de alerta epidemiológico  

---

## 🧰 Tecnologias Utilizadas

Categoria           | Ferramentas  
------------------- | ------------------------------  
Linguagem           | Python  
Redes Neurais       | TensorFlow, Keras  
Pré-processamento   | Pandas, NumPy, Scikit-learn, Joblib  
Visualização        | Matplotlib, Seaborn  

---


## 🤝 Agradecimentos

O ChatGPT apoiou este projeto na:

- transição do modelo de regressão para classificação,  
- construção da arquitetura ANN,  
- padronização do pré-processamento,  
- criação dos scripts de plotagem,  
- organização final deste README.  

---

## ✨ Autor

👨‍💻 **Vitor Leal**  

