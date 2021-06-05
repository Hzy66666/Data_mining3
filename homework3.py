import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('vgsales.csv')
# print(df.head(10))
df.dropna(inplace=True)


def top10(feature):
    if feature == 'Games':
        game_df = df.sort_values('Global_Sales', ascending=False, inplace=False)[:10]
        sales = game_df.groupby(['Name'])['Global_Sales'].sum().sort_values(ascending=False)
    else:
        sales = df.groupby([feature])['Global_Sales'].sum().sort_values(ascending=False)[:10]
    plt.figure(figsize=(12, 6))
    sns.barplot(y=sales.index, x=sales.values)
    plt.xticks(fontsize=15)
    plt.xlabel('Global Sales', fontsize=17)
    plt.yticks(fontsize=15)
    plt.ylabel(feature, fontsize=17)
    plt.title('Global TOP10 Most Popular ' + feature, fontsize=17)
    plt.savefig('img/' + feature + 'top10.jpg')
    plt.show()

top10('Games')
top10('Genre')
top10('Platform')
top10('Publisher')

data = df.copy()
data.dropna(inplace=True)
data.drop(['Name', 'Rank'], axis=1, inplace=True)
data['Platform'] = LabelEncoder().fit_transform(data['Platform'].astype('str'))
data['Genre'] = LabelEncoder().fit_transform(data['Genre'].astype('str'))
data['Publisher'] = LabelEncoder().fit_transform(data['Publisher'].astype('str'))
# print(data.head(10))

corr = data.corr()
print(corr)
plt.figure(figsize=(15, 15))
ax = sns.heatmap(corr, annot=True)

features = ['Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales']
X = data[features]
Y = data.Global_Sales 
train_X , test_X , train_Y , test_Y = train_test_split(X, Y)
scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
model = LinearRegression()
model.fit(train_X, train_Y)
predictions = model.predict(test_X)
error = mean_squared_error(predictions, test_Y)
print('loss: '+ str(error))

compare = pd.DataFrame({'Prediction': predictions, 'Label': test_Y}).head(10)
compare.plot(kind='barh')
plt.show()

years_sales = df.groupby(['Year'])['Global_Sales'].sum()
plt.figure(figsize=(12, 6))
sns.barplot(x=years_sales.index, y=years_sales.values)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=15)
plt.ylabel('Global Sales', fontsize=15)
plt.title('Global Video Game Sales Trend', fontsize=15)
plt.show()