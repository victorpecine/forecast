import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler # Normalização dos dados
from sklearn.model_selection import train_test_split # Separação em dados de teste e treino
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
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



# Dataframe
query = 'SELECT * FROM sensor_af2.df_af2 WHERE evento < 14 AND passoatual = 0'

df = pd.read_sql(query, mydb)

correlacao = df[['pressure', 'temperature', 'volume', 'volumetricflowrate']].corr()
print(correlacao)
"""
                    pressure  temperature    volume  volumetricflowrate
pressure            1.000000     0.320821 -0.370697           -0.505431
temperature         0.320821     1.000000 -0.986516           -0.685875
volume             -0.370697    -0.986516  1.000000            0.769641
volumetricflowrate -0.505431    -0.685875  0.769641            1.000000
"""

X = df[['volume']]
y = df['volumetricflowrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=113)

# Normalização dos dados
escala = StandardScaler()

X_train = escala.fit_transform(X_train)
X_test = escala.fit_transform(X_test)

df_train = pd.DataFrame(data=X_train, columns=['volume'])
df_train['volumetricflowrate'] = y_train

# Gradiente descendente
grad_descend = SGDRegressor()
grad_descend.fit(X_train, y_train)

y_pred = grad_descend.predict(X_test)

ax = sns.scatterplot(x=y_test, y=y_pred)
plt.show()
