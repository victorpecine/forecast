import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Normalização dos dados
from sklearn.model_selection import train_test_split # Separação em dados de teste e treino
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
from sklearn.metrics import r2_score # Coeficiente de determinação
from sqlalchemy import create_engine
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_error)
from lazypredict.Supervised import LazyRegressor
from math import sqrt



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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# print(models)
"""
                               Adjusted R-Squared  R-Squared     RMSE  Time Taken
Model
Lars                                         0.97       0.97  4834.26        0.01
LinearRegression                             0.97       0.97  4834.26        0.01
TransformedTargetRegressor                   0.97       0.97  4834.26        0.02
LassoLarsCV                                  0.97       0.97  4834.26        0.01
RANSACRegressor                              0.97       0.97  4834.26        0.01
LassoLarsIC                                  0.97       0.97  4834.26        0.02
LarsCV                                       0.97       0.97  4834.26        0.03
OrthogonalMatchingPursuit                    0.97       0.97  4834.26        0.00
SGDRegressor                                 0.97       0.97  4834.30        0.02 <-
Lasso                                        0.97       0.97  4834.39        0.00
LassoLars                                    0.97       0.97  4834.86        0.01
LassoCV                                      0.97       0.97  4837.45        0.06
RidgeCV                                      0.97       0.97  4851.16        0.02
HuberRegressor                               0.97       0.97  5105.89        0.03
Ridge                                        0.97       0.97  5173.86        0.00
ExtraTreeRegressor                           0.96       0.96  5867.47        0.01
ExtraTreesRegressor                          0.96       0.96  5883.00        0.10
AdaBoostRegressor                            0.94       0.95  6639.33        0.10
RandomForestRegressor                        0.94       0.95  6764.79        0.15
BaggingRegressor                             0.94       0.95  6899.99        0.02
PoissonRegressor                             0.93       0.94  7327.41        0.01
KNeighborsRegressor                          0.93       0.94  7343.67        0.02
GradientBoostingRegressor                    0.92       0.93  8029.65        0.06
XGBRegressor                                 0.92       0.93  8150.33        0.08
DecisionTreeRegressor                        0.92       0.93  8150.52        0.02
ElasticNet                                   0.82       0.84 11963.77        0.02
TweedieRegressor                             0.64       0.68 16892.26        0.01
GammaRegressor                               0.64       0.68 16950.92        0.02
ElasticNetCV                                -0.22      -0.07 31027.84        0.07
NuSVR                                       -0.24      -0.09 31345.21        0.03
BayesianRidge                               -0.32      -0.15 32271.16        0.12
DummyRegressor                              -0.32      -0.15 32271.16        0.00
LGBMRegressor                               -0.32      -0.15 32271.16        0.07
HistGradientBoostingRegressor               -0.32      -0.15 32271.16        0.08
SVR                                         -0.64      -0.43 35978.11        0.00
QuantileRegressor                           -0.64      -0.44 35985.03        0.05
PassiveAggressiveRegressor                  -5.21      -4.44 70029.25        0.02
KernelRidge                                 -5.49      -4.68 71561.41        0.03
GaussianProcessRegressor                    -6.51      -5.58 77025.63        0.05
LinearSVR                                   -9.13      -7.86 89434.16        0.00
MLPRegressor                                -9.13      -7.87 89446.59        0.07
"""

# Normalização dos dados
escala = StandardScaler()
X_norm = escala.fit_transform(X)

X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=0)


# Gradiente descendente
epocas = 1000
alpha = 0.01 # Taxa de aprendizagem
tolerancia = 0.0001


modelo = SGDRegressor(max_iter=epocas,
                      eta0=alpha,
                      tol=tolerancia,
                      verbose=0)

modelo.fit(X_norm_train, y_train)

# Previsão
y_prev = modelo.predict(X_norm_train)
# print(y_prev)


# 'Desfazendo' a normalização
X_test = escala.inverse_transform(X_norm_test)

X_train = escala.inverse_transform(X_norm_train)


plt.scatter(x=X_train, y=y_train, c='blue', marker='*')

plt.scatter(x=X_train, y=y_prev, c='red', marker='o')

plt.legend(labels=['y real', 'y previsto'])

plt.show()

# Modelo
sgdr = SGDRegressor()
sgdr.fit(X_norm_train, y_train)

# Métricas do modelo
# O R-quadrado (coeficiente de determinação) representa qual a porcentagem de dados que são explicados pelo modelo
r_squared = r2_score(y_train, y_prev)
# print(round(r_squared, 2))
# 0.94

# Root Mean Squared Erros
# O valor indica a raiz da distância entre o valor predito e o real
rmse = sqrt(mean_squared_error(y_train, y_prev))
# print(round(rmse, 2))
# 5925.89

# Mean Absolute Error
# O valor indica qual a margem de erro do modelo para cima ou para baixo
mae = mean_absolute_error(y_train, y_prev)
# print(round(mae, 2))
# 5091.18


anos_experiencia = 2
X_pontual = [[anos_experiencia]]

y_pontual_prev = sgdr.predict(X_pontual)[0]
print(round(y_pontual_prev, 2))
