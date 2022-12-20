import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler # Normalização dos dados
from sklearn.model_selection import train_test_split # Separação em dados de teste e treino
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
from sklearn.metrics import accuracy_score # Avaliação do modelo
from sklearn.metrics import r2_score # Coeficiente de determinação


# Regressão Linear com Gradiente Descendente -  Uma variável
# Acesso ao banco de dados
user = 'root'
password = 'password'
host = '127.0.0.1'
port = 3310
database = 'sensor_af2'

mydb = create_engine(url='mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                     user, password, host, port, database))



passo = 0

# Dataframe
query = 'SELECT * FROM sensor_af2.df_af2'

df = pd.read_sql(query, mydb)

# Pré-processamento de dados
X = df[['evento']]
y = df['volumetricflowrate']

# Normalização dos dados
escala = StandardScaler()
X_norm = escala.fit_transform(X)

X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=243)

# Gradiente descendente
epocas = 500
alpha = 0.01 # Taxa de aprendizagem
tolerancia = 0.01


modelo = SGDRegressor(max_iter=epocas,
                      eta0=alpha,
                      tol=tolerancia,
                      verbose=1)


# Treinamento
reg_linear = modelo.fit(X_norm_train, y_train)

# Previsão
y_prev = reg_linear.predict(X_norm_test)

plt.scatter(x=X_norm_test, y=y_test, c='blue', marker='*')

plt.scatter(X_norm_test, y_prev, c='red', marker='o')

plt.legend(labels=['y real', 'y previsto'])
plt.show()

# 'Desfazendo' a normalização
X_test = escala.inverse_transform(X_norm_test)

plt.scatter(x=X_test, y=y_test, c='blue', marker='*')

plt.scatter(X_test, y_prev, c='red', marker='o')

plt.legend(labels=['y real', 'y previsto'])
plt.show()


# Métricas do modelo
accuracy = r2_score(y_test, y_prev)
print(accuracy)

sgdr = SGDRegressor()
sgdr.fit(X_norm_train, y_train)

# O R-quadrado (coeficiente de determinação) é uma medida estatística de quão próximos os dados estão da linha de regressão ajustada
r_squared = sgdr.score(X_norm_train, y_train)
print(r_squared)
