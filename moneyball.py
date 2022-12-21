import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression


# Source: https://towardsdatascience.com/moneyball-linear-regression-76034259af5e

df = pd.read_csv('dataset/moneyball.csv', sep=',', encoding='utf-8')
"""
RS — Runs Scored
RA — Runs Allowed
W — Wins
OBP — On Base Percentage
SLG — Slugging Percentage
BA — Batting Average
Playoffs — Whether a team made it to playoffs or not
OOBP — Opponent’s On Base Percentage
OSLG — Opponents Slugging Percentage
"""

# RD, which shows the difference between RS(Runs Scored) and RA(Runs Allowed)
df['RD'] = df['RS'] - df['RA']

# Make sure we only have pre-Moneyball data
df = df[df['Year'] < 2002]

# Exploratory Data Analysis (EDA)
# Defining a colour palette
flatui = ['#6cdae7', '#fd3a4a', '#ffaa1d', '#ff23e5', '#34495e', '#2ecc71']
sns.set_palette(flatui)
# sns.palplot(sns.color_palette()) # Just plot the palette

# Run Difference vs Wins
sns.scatterplot(x='W', y='RS', hue='Playoffs', data=df)
plt.xlabel('Wins', fontsize=20)
plt.ylabel('Runs Scored', fontsize=20)
plt.axvline(99, 0, 1, color='Black', ls='--')
# plt.show()
# As there has only ever been 3 instances where a team has not made playoffs with ≥ 99 wins

# Relationship between Run Difference and Wins
x = np.array(df.RD)
y = np.array(df.W)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Slope = %s" % (slope), fontsize = 12)
plt.xlabel("Run Difference", fontsize =20)
plt.ylabel("Wins", fontsize = 20)
plt.axhline(99, 0, 1, color = "k", ls = '--')
# plt.show()

correlation = df[['RD', 'W', 'Playoffs']].corr(method='pearson')
"""
                RD         W  Playoffs
RD        1.000000  0.938515  0.566274
W         0.938515  1.000000  0.588978
Playoffs  0.566274  0.588978  1.000000

Wins and Run Difference has an extremely strong relationship
"""

correlation= df[['OBP','SLG','BA','RS']].corr(method='pearson')
"""
          OBP       SLG        BA        RS
OBP  1.000000  0.806154  0.854055  0.904909
SLG  0.806154  1.000000  0.814068  0.926384
BA   0.854055  0.814068  1.000000  0.831625
RS   0.904909  0.926384  0.831625  1.000000
"""

# Using univariate selection to select those features that have the strongest relationship with the output variable (RD)
# SelectKBest class allows us to pick a specific number of features
# We will use the chi-squared statistical test for non-negative features to select the best features from our dataset
df = df.dropna()

X = df.iloc[:, 6:9]
y = df.iloc[:, -1]

# Apply SelectKBest class to get best features
bestfeatures = SelectKBest(score_func=chi2, k=3)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores.columns = ['Feature','Score']
# print(featureScores.nlargest(3,'Score'))
"""
Feature     Score
1     SLG  0.008951
0     OBP  0.006262
2      BA  0.004577
"""

# Model Building
# Build a model for Runs Scored, predicted using On Base Percentage and Slugging Percentage
x = df[['OBP','SLG']].values
y = df[['RS']].values
Runs = LinearRegression()
Runs.fit(x,y)

intercept = Runs.intercept_[0]
# -1063.22544751
obp = Runs.coef_[0][0]
# 2724.40798084
slg = Runs.coef_[0][1]
# 2190.91660661

# Build a model for Runs Allowed, using Opponents On Base Percentage and Opponents Slugging Percentage
x = df[['OOBP','OSLG']].values
y = df[['RA']].values
RunsAllowed = LinearRegression()
RunsAllowed.fit(x,y)

intercept = RunsAllowed.intercept_[0]
# -775.1623486135549
oobp = RunsAllowed.coef_[0][0]
# 3225.00378812453
oslg = RunsAllowed.coef_[0][1]
# 1106.5039097519152


# Model to predict the number of Wins when given Run Difference.

x = df[['RD']].values
y = df[['W']].values
Wins = LinearRegression()
Wins.fit(x,y)

intercept = Wins.intercept_[0]
# 84.09190384923886
rd = Wins.coef_[0][0]
# 0.08469338646898401

"""
Predictions
The A’s 2002 team pre-season statistics taken from 2001
"""

OBP  = 0.339
SLG  = 0.430
OOBP = 0.307
OSLG = 0.373

pred_runs_scored = round(Runs.predict([[OBP, SLG]])[0][0], 0)
# print(pred_runs_scored)
# 802.0

pred_runs_allowed = round(RunsAllowed.predict([[OOBP, OSLG]])[0][0], 0)
# print(pred_runs_allowed)
# 627.0

pred_rd = pred_runs_scored - pred_runs_allowed
# print(pred_rd)
# 174.0

pred_wins = round(Wins.predict([[pred_rd]])[0][0], 0)
# print(pred_wins)
# 99.0
