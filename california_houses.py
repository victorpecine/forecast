import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Separação em dados de teste e treino
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
from sklearn.metrics import r2_score # Coeficiente de determinação
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm


california_housing = fetch_california_housing(as_frame=True)
df = fetch_california_housing(as_frame=True).frame
df = pd.DataFrame(df)

correlacao = df[['MedInc', 'AveRooms', 'AveBedrms', 'AveOccup', 'Population', 'MedHouseVal']].corr()
# print(correlacao)
"""
               MedInc  AveRooms  AveBedrms  AveOccup  Population  MedHouseVal
MedInc       1.000000  0.326895  -0.062040  0.018766    0.004834     0.688075
AveRooms     0.326895  1.000000   0.847621 -0.004852   -0.072213     0.151948
AveBedrms   -0.062040  0.847621   1.000000 -0.006181   -0.066197    -0.046701
AveOccup     0.018766 -0.004852  -0.006181  1.000000    0.069863    -0.023737
Population   0.004834 -0.072213  -0.066197  0.069863    1.000000    -0.024650
MedHouseVal  0.688075  0.151948  -0.046701 -0.023737   -0.024650     1.000000

AveRooms e AveBedrms possuem uma correlação de 0.847621
É necessário criar dois modelos, cada um com uma dessas variáveis
A alta correlação (> 0.5) pode interferir no resultado do modelo caso as duas variáveis estejam presentes
"""

# # Modelo sem AveRooms
# X = df[['MedInc', 'AveBedrms', 'AveOccup', 'Population']]
# y = df['MedHouseVal']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Normalização
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# # Gradiente descendente
# epocas = 3000
# alpha = 0.001 # Taxa de aprendizagem
# tolerancia = 0.0001


# modelo = SGDRegressor(max_iter=epocas,
#                       eta0=alpha,
#                       tol=tolerancia,
#                       verbose=0)

# modelo.fit(X_train, y_train)

# y_pred = modelo.predict(X_test)

# # mean_squared_error = round(sm.mean_squared_error(y_test, y_pred), 2)
# # print(mean_squared_error)
# # # 0.72

# # r2_scr = round(sm.r2_score(y_test, y_pred), 2)
# # print(r2_scr)
# # # 0.46

# df_pred = pd.DataFrame({'Actual Value': y_test,
#                         'Predicted Value': y_pred,
#                         'Difference': y_test - y_pred,
#                         'Difference_squared': (y_test - y_pred)**2})

# # ax = sns.regplot(x=df_pred.index,
# #                  y=df_pred['Difference'],
# #                  scatter_kws={"color": "blue", "s": 1},
# #                  line_kws={"color": "red", "lw": 2})
# # plt.show()

# # ax = sns.regplot(x=df_pred.index,
# #                  y=df_pred['Difference_squared'],
# #                  scatter_kws={"color": "blue", "s": 1},
# #                  line_kws={"color": "red", "lw": 2})
# # plt.show()

# # Preditor de Valor da casa
# X_pontual = X_test[0:1]
# """
# MedInc     = 0.14156681
# AveBedrms  = -0.04931106   
# AveOccup   = 0.06581932  
# Population = 0.11432005
# """

# y_pontual = modelo.predict(X_pontual)[0]
# # print(y_pontual)
# # 2.176988066557279







# Modelo sem AveBedrms
X = df[['MedInc', 'AveRooms', 'AveOccup', 'Population']]
y = df['MedHouseVal']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Gradiente descendente
epocas = 3000
alpha = 0.001 # Taxa de aprendizagem
tolerancia = 0.0001


modelo = SGDRegressor(max_iter=epocas,
                      eta0=alpha,
                      tol=tolerancia,
                      verbose=0)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mean_squared_error = round(sm.mean_squared_error(y_test, y_pred), 2)
print(mean_squared_error)
# 0.72

r2_scr = round(sm.r2_score(y_test, y_pred), 2)
print(r2_scr)
# 0.46

df_pred = pd.DataFrame({'Actual Value': y_test,
                        'Predicted Value': y_pred,
                        'Difference': y_test - y_pred,
                        'Difference_squared': (y_test - y_pred)**2})

# ax = sns.regplot(x=df_pred.index,
#                  y=df_pred['Difference'],
#                  scatter_kws={"color": "blue", "s": 1},
#                  line_kws={"color": "red", "lw": 2})
# plt.show()

# ax = sns.regplot(x=df_pred.index,
#                  y=df_pred['Difference_squared'],
#                  scatter_kws={"color": "blue", "s": 1},
#                  line_kws={"color": "red", "lw": 2})
# plt.show()

# Preditor de Valor da casa
X_pontual = X_test[0:1]
print(X_pontual)
"""
MedInc     = 0.14156681
AveRooms   = 0.07721063   
AveOccup   = 0.06581932  
Population = 0.11432005
"""

y_pontual = modelo.predict(X_pontual)[0]
print(y_pontual)
# 2.1715110679495653
