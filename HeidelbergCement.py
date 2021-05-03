# HeidelbergCement Material Strength

# -------------------------------------------------------------------------------------------------
# Import ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

path = "C:/Users/HP/OneDrive/Documents/Python Anaconda/Heidelberg Cement"
os.chdir(path)

data = pd.read_excel('interview_dataset.xlsx', sheet_name='Sheet1')
data.head()

# Exploratory analysis

# -------------------------------------------------------------------------------------------------
# Summary -----------------------------------------------------------------------------------------

data.describe()

# -------------------------------------------------------------------------------------------------
# NAs ---------------------------------------------------------------------------------------------

plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

# -------------------------------------------------------------------------------------------------
# Correlation -------------------------------------------------------------------------------------

data.corr()

plt.figure(figsize=(16, 10))
sns.heatmap(data.corr())
plt.show

plt.figure(figsize=(18, 10))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

# -------------------------------------------------------------------------------------------------
# Distribution ------------------------------------------------------------------------------------

data.hist(bins=30, figsize=(16, 15))
plt.show()

sns.set_style('whitegrid') # selecting a particular variable
sns.histplot(data['X_17'], bins=20) # CHANGE
plt.show()

# Feature engineering

# -------------------------------------------------------------------------------------------------
# Feature engineering preprocessing ---------------------------------------------------------------

data['X_26'].value_counts() # X_26 separate time and date
data['time'] = pd.to_datetime(data['X_26'], format='%Y:%M:%D').dt.time
data['date'] = pd.to_datetime(data['X_26'], format='%Y:%M:%D').dt.date
data = data.drop('X_26', 1)
data.head()

import plotly.express as px # material strength over time
fig = px.scatter(data, x='date', y='y',
                 title="Material strength (y) through the time",
                 template="simple_white")
fig.show()

data['X_28'].value_counts() # X_28 from a string into two categories
data[['X_28','X_29']] = data.X_28.str.split("---",expand=True,)
data = data[['X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9',
       'X_10', 'X_11', 'X_12', 'y', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18',
       'X_19', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'date',  'time', 'X_28',
       'X_29']]
data.head()

import plotly.express as px # separation of trends based on one category
fig = px.scatter(data, x='date', y='y', facet_row="X_29",
                 title="Material strength (y) through the time",
                 template="simple_white")
fig.show()

import plotly.express as px # separation of trends based on one category
fig = px.scatter(data, x='date', y='y', facet_col="X_29",
                 title="Material strength (y) through the time",
                 template="simple_white")
fig.show()

data['X_14'].value_counts() # X_14 deleting of a column with NAs only
data = data.drop('X_14', 1)
data.head()

data['X_12'].value_counts() # X_12 simple linear regression for imputation
X_12 = data[['X_11', 'X_12']]
X_12 = X_12.dropna(how = 'any')

from sklearn.model_selection import train_test_split
X = X_12[['X_11']]
y = X_12['X_12']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1001)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
model = LinearRegression() # linear regression for getting the coefficient
model.fit(X_train,y_train)
coefficient = pd.DataFrame(model.coef_,X.columns)
coefficient.columns = ['Coefficient']
coefficient = pd.DataFrame(coefficient)

coef = round(coefficient['Coefficient'][0],4)
inter = round(model.intercept_,4)
y_pred = model.predict(X_test)
print('Coefficient: ', round(coefficient['Coefficient'][0],4))
print('Intercept: ', round(model.intercept_,4))
print('Mean absolute error: ',round(mean_absolute_error(y_test, y_pred),4))

plt.figure(figsize=(12,6)) # visualisation of regression for imputation
plt.scatter(X, y, alpha=0.9)

x = np.array(X_12.X_11)

# Fit function
f = lambda x: coef*x + inter
# Plot fit
plt.plot(x,f(x),lw=2.5, c="r", label="predicted X_12")
plt.xlabel("X_11")
plt.ylabel("X_12")
plt.legend()
plt.show()

def impute_X_12(cols): # imputation with help of the coefficient
    X_11 = cols[0]
    X_12 = cols[1]

    if pd.isnull(X_12):
        return X_11 * coef
    else:
        return X_12

data['X_12'] = round(data[['X_11','X_12']].apply(impute_X_12, axis=1),1)
data['X_12'].count()

data['X_0'].value_counts() # X_0 deleting of a column with lot of NAs
data = data.drop('X_0', 1)
data['X_12'].head()

data['X_17'].value_counts() # X_17 mean imputation
mean = data['X_17'].mean()
mean

def impute_X_17(cols):
    X_17 = cols[0]

    if pd.isnull(X_17):
        return mean
    else:
        return X_17

data['X_17'] = round(data[['X_17']].apply(impute_X_17, axis=1),1)
data['X_17'].count()

# REST of NAs deleted
data = data.dropna()
data

# Models

# -------------------------------------------------------------------------------------------------
# Model1 - Multiple Linear Regression X -----------------------------------------------------------

data.columns # splitting data into train and test set
df = data[['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10',
       'X_11', 'X_12', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_28', 'X_29', 'y']]

from sklearn.model_selection import train_test_split
X = df[['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_7', 'X_8', 'X_9', 'X_10',
       'X_11', 'X_12', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_28', 'X_29']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1001)

from sklearn.linear_model import LinearRegression
model1 = LinearRegression() # multiple linear regression model
model1.fit(X_train, y_train)
coefficient = pd.DataFrame(model1.coef_,X.columns)
coefficient.columns = ['Coefficient']
coefficient = pd.DataFrame(coefficient)

y_pred = model1.predict(X_test)

outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)
print('Percentage difference: ', round(outcome.difference_percentage.abs().mean(),2),'%')

coef = round(coefficient['Coefficient'][0],4)
inter = round(model1.intercept_,4)
y_pred = model1.predict(X_test)
print('Coefficients: ','\n', round(coefficient['Coefficient'],4))

print('Intercept: ', round(model1.intercept_,4))
print('Mean absolute error: ',round(mean_absolute_error(y_test, y_pred),4))

# -------------------------------------------------------------------------------------------------
# Model2 - Multiple Linear Regression y error -----------------------------------------------------

data.columns # splitting data into train and test set
df = data[['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y']]

df['e_0'] = df['y_0'] - df['y']
df['e_1'] = df['y_1'] - df['y']
df['e_2'] = df['y_2'] - df['y']
df['e_3'] = df['y_3'] - df['y']
df['e_4'] = df['y_4'] - df['y']
df['e_5'] = df['y_5'] - df['y']

df['e'] = df[['e_0', 'e_1', 'e_2', 'e_3', 'e_4', 'e_5']].mean(axis=1)
df
from sklearn.model_selection import train_test_split
X = df[['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5']]
y = df['e']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1001)

from sklearn.linear_model import LinearRegression
model2 = LinearRegression() # multiple linear regression model
model2.fit(X_train, y_train)
coefficient = pd.DataFrame(model2.coef_,X.columns)
coefficient.columns = ['Coefficient']
coefficient = pd.DataFrame(coefficient)

y_pred = model2.predict(X_test)

coef = round(coefficient['Coefficient'][0],4)
inter = round(model2.intercept_,4)
y_pred = model2.predict(X_test)
print('Coefficients: ','\n', round(coefficient['Coefficient'],4))

print('Intercept: ', round(model2.intercept_,4))
print('Mean absolute error: ',round(mean_absolute_error(y_test, y_pred),4))

# -------------------------------------------------------------------------------------------------
# Model3 - Random Forest Regression X -------------------------------------------------------------

data.columns # splitting data into train and test set
df = data[['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10',
       'X_11', 'X_12', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_28', 'X_29', 'y']]

from sklearn.model_selection import train_test_split
X = df[['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_7', 'X_8', 'X_9', 'X_10',
       'X_11', 'X_12', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_28', 'X_29']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1001)

from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor(n_estimators=500, random_state=1001) # random forest regression model
model3.fit(X_train, y_train)

y_pred = model3.predict(X_test)

outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)
print('Percentage difference: ', round(outcome.difference_percentage.abs().mean(),2),'%')

print('Mean absolute error: ',round(mean_absolute_error(y_test, y_pred),4))

# Conclusion

### The random forest model performed the best out of the models that I tried: decision tree regression, multiple linear regression, SVM and ANN. The random forest does not take into consideration outliers and even in comparison to the linear regression model where is high linear dependency, performs very well.

### The Linear regression model of y error shows a really low mean absolute error because we try to estimate the error out of the six measurements. One can see high linear dependency, which resulted in a choice of multiple linear regression model.

# ADDITIONAL NOTES --------------------------------------------------------------------------------

# Decision Tree

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1001)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)
print('Percentage difference: ', round(outcome.difference_percentage.abs().mean(),2),'%')

print('Mean absolute error: ',round(mean_absolute_error(y_test, y_pred),4))

# SVM

X = X_train.values
y = y_train.values
y = np.array(y).reshape(len(y),1)
print(X); print(y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
model = SVR(kernel='rbf')
model.fit(X, y)

y_pred = sc_y.inverse_transform(model.predict(sc_X.transform(X_test.values)))
print('Mean absolute error: ',round(mean_absolute_error(y_test, y_pred),4))

# ANN

# Spliting
X = df.drop('y', axis=1).values # Values are important for the model
y = df['y'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model preparation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

print(X_train.shape) # Number of neurons

# Model
model = Sequential()
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Model fitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, y=y_train, 
         validation_data=(X_test, y_test),
         batch_size=128, epochs=168, callbacks=[early_stop]) # batch_size agains overfitting

losses = pd.DataFrame(model.history.history)
losses.plot(); plt.show()

# Model errors
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test)
print('Mean squared error: ',round(np.sqrt(mean_squared_error(y_test, predictions)),2))
print('Mean absolute error: ',round(mean_absolute_error(y_test, predictions),2))
print('Explained variance: ',round(explained_variance_score(y_test, predictions),2))


