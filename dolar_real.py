import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from lazypredict.Supervised import LazyRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose




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
X = df.index
y = df['valor_real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Autocorrelação
# x = pd.plotting.autocorrelation_plot(df)
# x.plot()
# plt.show()

# Salvar a decomposicao em result
result = seasonal_decompose(df, period=1)

# Plotar os 4 gráficos
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(12,8))
result.observed.plot(ax=ax1) # Observação original
result.trend.plot(ax=ax2) # Tendência
result.seasonal.plot(ax=ax3) # Sazonalidade
result.resid.plot(ax=ax4) # Residual

plt.tight_layout()
plt.show()
