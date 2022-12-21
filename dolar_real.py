import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split



user = 'admin'
password = 'password'
host = '127.0.0.1'
port = 3312
database = 'time_series'

mydb = create_engine(url='mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                     user, password, host, port, database))


# # Dataframe
# query = 'SELECT * FROM time_series.dolar_real;'

# df = pd.read_sql(query, mydb)

# # Conversão para datetime64[ns]
# df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

# # Enviando a alteração para o banco
# df.to_sql(name='dolar_real', if_exists='replace', con=mydb, index=False)


# Dataframe
query = 'SELECT * FROM time_series.dolar_real;'

df = pd.read_sql(query, mydb)

df.index = df['data']
df = df.drop(columns=['id', 'data'])

# ax = sns.lineplot(data=df, y='valor_real', x='data')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Relação R\$ x US\$',
#        xlabel=None,
#        ylabel=None)
# plt.show()


# Separação dos dados em teste e treino
# treino = df[df.index < pd.to_datetime('2013-09-16', format='%Y-%m-%d')]
# teste = df[df.index > pd.to_datetime('2013-09-16', format='%Y-%m-%d')]

X = df['valor_real']
y = df.index

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
Método Autoregressive Moving Average (ARMA)

To define an ARMA model with the SARIMAX class, we pass in the order parameters of (1, 0 ,1). Alpha corresponds to the significance level of our predictions. Typically, we choose an alpha = 0.05. Here, the ARIMA algorithm calculates upper and lower bounds around the prediction such that there is a 5 percent chance that the real value will be outside of the upper and lower bounds. This means that there is a 95 percent confidence that the real value will be between the upper and lower bounds of our predictions.
https://builtin.com/data-science/time-series-forecasting-python
"""

ARMAmodel = SARIMAX(y_train, order = (1, 0, 1))
ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(X_test))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
print(y_pred_df)
# y_pred_df.index = X_test
# y_pred_out = y_pred_df["Predictions"] 

