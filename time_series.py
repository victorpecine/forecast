import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine



# Acesso ao banco de dados
user = 'admin'
password = 'password'
host = '127.0.0.1'
port = 3312
database = 'time_series'



mydb = create_engine(url='mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                     user, password, host, port, database))

query = "SELECT * FROM power_usage WHERE date LIKE '2017-01%%'"

df = pd.read_sql(query, mydb)

# Nome do dia da semana
# df['week_day'] = df['date'].dt.day_name().str.lower()
# df.to_sql(name='power_usage', con=mydb, if_exists='replace')


# ax = sns.lineplot(data=df, y='kWh', x='date')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Power usage',
#        xlabel=None,
#        ylim=(0))
# plt.show()

# ax = sns.scatterplot(data=df, y='kWh', x='date')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Power usage',
#        xlabel=None,
#        ylim=(0))
# plt.show()

# ax = sns.lmplot(data=df, y='kWh', x='day_of_week')
# ax.figure.set_size_inches(18,6)
# ax.set(title='Power usage',
#        xlabel=None,
#        ylim=(0))
# plt.show()

# ax = sns.boxplot(data=df, x='kWh')
# plt.show()
