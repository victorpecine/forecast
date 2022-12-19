import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Normalização dos dados
from sklearn.model_selection import train_test_split # Separação em dados de teste e treino
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
from sklearn.metrics import accuracy_score # Avaliação do modelo
from sklearn.metrics import r2_score # Coeficiente de determinação


# Regressão Linear com Gradiente Descendente
# PRevisão da autonomia do veículo (km/l) a partir do peso
df = pd.read_csv('dataset/auto-mpg.csv', sep=',', encoding='utf-8')

# Análise gráfica com linha de tendência
# ax = sns.regplot(data=df, x='weight', y='mpg')
# ax.figure.set_size_inches(18,6)
# ax.set(xlabel='Peso (lb)',
#        ylim=0,
#        ylabel='Autonomia (mpg)')
# plt.show()


# Pré-processamento de dados
df['peso_kg'] = df['weight'] * 0.453592
df['km/l'] = df['mpg'] * 0.425144

X = df[['peso_kg']]
y = df['km/l']

# Normalização dos dados
escala = StandardScaler()
X_norm = escala.fit_transform(X)

X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.15, random_state=243)

# Gradiente descendente
epocas = 500
alpha = 0.01 # Taxa de aprendizagem
tolerancia = 0.0001


modelo = SGDRegressor(max_iter=epocas,
                      eta0=alpha,
                      tol=tolerancia,
                      verbose=0)


# Treinamento
reg_linear = modelo.fit(X_norm_train, y_train)

# Previsão
y_prev = reg_linear.predict(X_norm_test)

# plt.scatter(x=X_norm_test, y=y_test, c='blue', marker='*')

# plt.scatter(X_norm_test, y_prev, c='red', marker='o')

# plt.legend(labels=['y real', 'y previsto'])
# plt.show()

# 'Desfazendo' a normalização
X_test = escala.inverse_transform(X_norm_test)

# plt.scatter(x=X_test, y=y_test, c='blue', marker='*')

# plt.scatter(X_test, y_prev, c='red', marker='o')

# plt.legend(labels=['y real', 'y previsto'])
# plt.show()


# # Métricas do modelo
# accuracy = r2_score(y_test, y_prev)
# print(accuracy)
# # 0.7598937475878289

# sgdr = SGDRegressor()
# sgdr.fit(X_norm_train, y_train)

# # O R-quadrado (coeficiente de determinação) é uma medida estatística de quão próximos os dados estão da linha de regressão ajustada
# r_squared = sgdr.score(X_norm_train, y_train)
# print(r_squared)
# 0.6789768946804539
