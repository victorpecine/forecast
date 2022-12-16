import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('dataset/dolar.csv', sep=',')

ax = sns.lineplot(data=df)
ax.figure.set_size_inches(18,6)
ax.set(title='Relação R\$ x US\$',
       xlabel=None,
       ylabel=None)
plt.show()
