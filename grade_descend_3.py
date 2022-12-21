import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Normalização dos dados
from sklearn.model_selection import train_test_split # Separação em dados de teste e treino
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
from sklearn.metrics import r2_score # Coeficiente de determinação
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error 



# Regressão Linear com Gradiente Descendente
user = 'admin'
password = 'password'
host = '127.0.0.1'
port = 3312
database = 'time_series'

mydb = create_engine(url='mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                     user, password, host, port, database))


# Dataframe
query = 'SELECT * FROM time_series.salary_data'

df = pd.read_sql(query, mydb)

# # Análise gráfica com linha de tendência
# ax = sns.regplot(data=df, x='years_experience', y='salary')
# ax.figure.set_size_inches(18,6)
# ax.set(xlabel='Anos de experiência',
#        ylim=0,
#        ylabel='Salário (US\$)')
# plt.show()

X = df[['years_experience']]
y = df['salary']

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

plt.scatter(x=X_test, y=y_test, c='blue', marker='*')

plt.scatter(x=X_test, y=y_prev, c='red', marker='o')

plt.legend(labels=['y real', 'y previsto'])
plt.show()


# # Métricas do modelo
# sgdr = SGDRegressor()
# sgdr.fit(X_norm_train, y_train)

# # O R-quadrado (coeficiente de determinação) é uma medida estatística de quão próximos os dados estão da linha de regressão ajustada
# r_squared = sgdr.score(X_norm_train, y_train)
# print(r_squared)
# 0.9561175769700393


anos_experiencia = 2.2
X_pontual = [[anos_experiencia]]

y_pontual_prev = reg_linear.predict(X_pontual)[0]
# print(y_pontual_prev)

y_real = df.loc[df['years_experience'] == anos_experiencia, 'salary']

# erro = y_real - y_pontual_prev
# erro = mean_squared_error(y, y_prev)
# print('Erro de R$ {}'.format(erro))
print(y_prev)
