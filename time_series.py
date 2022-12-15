import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine



# Acesso ao banco de dados
user = 'admin'
password = 'password'
host = '127.0.0.1'
port = 3312
database = 'time_series'



mydb = create_engine(url='mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                     user, password, host, port, database))

# df = pd.read_csv('dataset/power_usage_2016_to_2020.csv', sep=',')

# # Please note
# # power_usage file :-Please note StartDate format while copying to csv file has mismatching date formats.
# # StartDate 2016-06-01 00-00-00 (yyyy-mm-dd HH-M-S) to end date 2020-07-07 23:00:00
# n = df.shape[0]
# p1 = pd.Series(range(n), pd.period_range('2016-06-01 00:00:00', freq = '1H', periods = n))
# df['StartDate'] = p1.to_frame().index

# df['media_movel_hora'] = df['Value (kWh)'].rolling(2).mean()

# df['var_percent_hora'] = df['Value (kWh)'].pct_change()

# df['dif_hora'] = df['Value (kWh)'].diff()

# df.to_sql(name='power_usage', con=mydb, if_exists='replace')

query = 'SELECT * FROM power_usage WHERE data LIKE "2018-02%%"'

df = pd.read_sql(query, mydb)

# Análise gráfica do consumo
# ax = sns.regplot(data=df, y='consumo', x='index', color='red')
# ax.figure.set_size_inches(18,6)
# ax.set(xlabel=None,
#        ylabel=None)

# ax = sns.lineplot(data=df, y='consumo', x='index', color='blue')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Consumo (kWh) 2018-02',
#        xlabel=None,
#        ylabel=None)

# ax = sns.boxplot(data=df, y='consumo', color='orange')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Consumo (kWh) 2018-02',
#        xlabel=None,
#        ylabel=None)

# plt.show()

# Dados estatísticos do consumo
estat = df['consumo'].describe()

primeiro_quartil = estat['25%']
terceito_quartil = estat['75%']

limite_inferior = primeiro_quartil - 1.5 * (terceito_quartil - primeiro_quartil)
# -0.14812499999999995
# Como o valor mínimo (0.196000) é maior que o limite inferior calculado (-0.1481249), o limite inferior considerado será 0.196000

limite_superior = terceito_quartil + 1.5 * (terceito_quartil - primeiro_quartil)
# 0.960875
# Como o valor máximo (3.845000) é maior que o limite superior calculado (0.960875), o limite superior considerado será 0.960875



# Análise gráfica da diferença do consumo
# ax = sns.regplot(data=df, y='dif_hora', x='index', color='red')
# ax.figure.set_size_inches(18,6)
# ax.set(xlabel=None,
#        ylabel=None)

# ax = sns.lineplot(data=df, y='dif_hora', x='index', color='blue')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Diferença consumo (kWh) 2018-02',
#        xlabel=None,
#        ylabel=None)

# ax = sns.boxplot(data=df, y='dif_hora', color='orange')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Diferença consumo (kWh) 2018-02',
#        xlabel=None,
#        ylabel=None)

# plt.show()
