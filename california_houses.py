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
from lazypredict.Supervised import LazyRegressor
from math import sqrt




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

# Modelo sem AveRooms
X = df[['MedInc', 'AveBedrms', 'AveOccup', 'Population']]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# print(models)
"""
                               Adjusted R-Squared  R-Squared   RMSE  Time Taken
Model
HistGradientBoostingRegressor                0.61       0.61   0.72        0.94
GradientBoostingRegressor                    0.60       0.60   0.73        2.10
LGBMRegressor                                0.60       0.60   0.73        0.26
MLPRegressor                                 0.60       0.60   0.73        6.76
XGBRegressor                                 0.58       0.58   0.74        1.18
RandomForestRegressor                        0.58       0.58   0.75        6.86
NuSVR                                        0.58       0.58   0.75       19.74
ExtraTreesRegressor                          0.57       0.57   0.76        3.52
SVR                                          0.57       0.57   0.76       21.83
BaggingRegressor                             0.55       0.55   0.77        0.69
KNeighborsRegressor                          0.52       0.52   0.80        0.15
LassoCV                                      0.46       0.46   0.85        0.17
ElasticNetCV                                 0.46       0.46   0.85        0.18
RidgeCV                                      0.46       0.46   0.85        0.03
BayesianRidge                                0.46       0.46   0.85        0.11
Ridge                                        0.46       0.46   0.85        0.06
LassoLarsIC                                  0.46       0.46   0.85        0.08
LassoLarsCV                                  0.46       0.46   0.85        0.03
OrthogonalMatchingPursuitCV                  0.46       0.46   0.85        0.04
LarsCV                                       0.46       0.46   0.85        0.08
Lars                                         0.46       0.46   0.85        0.06
TransformedTargetRegressor                   0.46       0.46   0.85        0.02
LinearRegression                             0.46       0.46   0.85        0.01
OrthogonalMatchingPursuit                    0.46       0.46   0.85        0.02
AdaBoostRegressor                            0.39       0.39   0.90        0.62
PoissonRegressor                             0.39       0.39   0.91        0.02
TweedieRegressor                             0.35       0.35   0.93        0.05
GammaRegressor                               0.34       0.34   0.94        0.04
ElasticNet                                   0.21       0.21   1.03        0.02
ExtraTreeRegressor                           0.19       0.19   1.04        0.05
DecisionTreeRegressor                        0.19       0.19   1.04        0.13
Lasso                                       -0.00      -0.00   1.15        0.03
DummyRegressor                              -0.00      -0.00   1.15        0.02
LassoLars                                   -0.00      -0.00   1.15        0.01
HuberRegressor                              -0.04      -0.04   1.18        0.11
LinearSVR                                   -0.07      -0.07   1.20        0.69
PassiveAggressiveRegressor                  -0.55      -0.55   1.44        0.03
RANSACRegressor                             -1.09      -1.08   1.67        0.42
KernelRidge                                 -2.74      -2.74   2.23       35.25
SGDRegressor                             -9248.44   -9242.46 111.01        0.03
GaussianProcessRegressor                 -9737.66   -9731.37 113.91      103.23
"""

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

rmse = sqrt(sm.mean_squared_error(y_test, y_pred))
print(rmse)
# 0.8497067909154983

mae = sm.mean_absolute_error(y_test, y_pred)
print(mae)
# 0.6313654807899233

r2 = sm.r2_score(y_test, y_pred)
print(r2)
# 0.4586099850737725

# # Preditor de Valor da casa
# X_pontual = X_test[0:1]

# y_pontual = modelo.predict(X_pontual)[0]
# # print(y_pontual)







# # Modelo sem AveBedrms
# Modelo sem AveRooms
X = df[['MedInc', 'AveRooms', 'AveOccup', 'Population']]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# print(models)
"""
                               Adjusted R-Squared  R-Squared   RMSE  Time Taken
Model
HistGradientBoostingRegressor                0.64       0.64   0.69        1.02
GradientBoostingRegressor                    0.64       0.64   0.70        2.30
LGBMRegressor                                0.64       0.64   0.70        0.30
MLPRegressor                                 0.63       0.63   0.70        6.40
XGBRegressor                                 0.62       0.62   0.71        1.51
NuSVR                                        0.62       0.62   0.72       24.01
RandomForestRegressor                        0.61       0.61   0.72       13.80
SVR                                          0.61       0.61   0.72       29.28
ExtraTreesRegressor                          0.61       0.61   0.72        4.14
BaggingRegressor                             0.58       0.58   0.75        0.74
KNeighborsRegressor                          0.55       0.55   0.77        0.20
LassoCV                                      0.46       0.46   0.85        0.22
ElasticNetCV                                 0.46       0.46   0.85        0.24
OrthogonalMatchingPursuit                    0.46       0.46   0.85        0.02
RidgeCV                                      0.45       0.45   0.85        0.04
BayesianRidge                                0.45       0.45   0.85        0.07
Ridge                                        0.45       0.45   0.85        0.05
LassoLarsIC                                  0.45       0.45   0.85        0.09
LassoLarsCV                                  0.45       0.45   0.85        0.04
OrthogonalMatchingPursuitCV                  0.45       0.45   0.85        0.06
LarsCV                                       0.45       0.45   0.85        0.10
Lars                                         0.45       0.45   0.85        0.08
TransformedTargetRegressor                   0.45       0.45   0.85        0.02
LinearRegression                             0.45       0.45   0.85        0.02
AdaBoostRegressor                            0.44       0.44   0.87        0.50
PoissonRegressor                             0.39       0.39   0.90        0.07
TweedieRegressor                             0.35       0.35   0.93        0.05
GammaRegressor                               0.34       0.34   0.94        0.03
DecisionTreeRegressor                        0.27       0.27   0.98        0.16
ExtraTreeRegressor                           0.24       0.24   1.01        0.08
ElasticNet                                   0.21       0.21   1.03        0.03
Lasso                                       -0.00      -0.00   1.15        0.04
DummyRegressor                              -0.00      -0.00   1.15        0.02
LassoLars                                   -0.00      -0.00   1.15        0.02
HuberRegressor                              -0.31      -0.31   1.32        0.15
LinearSVR                                   -0.44      -0.44   1.38        0.40
PassiveAggressiveRegressor                  -0.58      -0.58   1.45        0.09
KernelRidge                                 -2.75      -2.74   2.23       39.38
RANSACRegressor                             -8.24      -8.24   3.51        1.28
SGDRegressor                            -10202.45  -10195.86 116.60        0.05
GaussianProcessRegressor                -14516.79  -14507.41 139.08      131.18
"""

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

rmse = sqrt(sm.mean_squared_error(y_test, y_pred))
print(rmse)
# 0.849592754692479

mae = sm.mean_absolute_error(y_test, y_pred)
print(mae)
# 0.6253197130771821

r2 = sm.r2_score(y_test, y_pred)
print(r2)
# 0.4586049149532281

# # Preditor de Valor da casa
# X_pontual = X_test[0:1]

# y_pontual = modelo.predict(X_pontual)[0]
# # print(y_pontual)
