#!/usr/bin/env python
# coding: utf-8

# # Predictive Modeling: Regression

# ![image.png](attachment:image.png)

# This project will entail employing Regression model(s) and evaluate its effectiveness in Predictive Modeling.
# 
# In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships 
# between a dependent variable (often called the 'outcome variable') and one or more independent variables 
# (often called 'predictors', 'covariates', or 'features'). 
# 
# Regression analysis is primarily used for two conceptually distinct purposes:
# 
# * First, regression analysis is widely used for prediction and forecasting, where its use has substantial overlap 
#   with the field of machine learning. 
# 
# * Second, in some situations regression analysis can be used to infer causal relationships between the independent 
#   and dependent variables. 
# 
# To use regressions for prediction or to infer causal relationships, respectively, a researcher must carefully justify 
# why existing relationships have predictive power for a new context or why a relationship between two variables has a 
# causal interpretation. The latter is especially important when researchers hope to estimate causal relationships using 
# observational data.
# 
# Source: Wikipedia

# ## Import Libraries

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/boston-house-prices/housing.csv', delim_whitespace=True, header=None)


# In[ ]:


df.head()


# | Code   | Description   |
# |:---|:---|
# |**CRIM** | per capita crime rate by town |
# |**ZN**  | proportion of residential land zoned for lots over 25,000 sq.ft. | 
# |**INDUS**  | proportion of non-retail business acres per town | 
# |**CHAS**  | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) | 
# |**NOX**  | nitric oxides concentration (parts per 10 million) | 
# |**RM**  | average number of rooms per dwelling | 
# |**AGE**  | proportion of owner-occupied units built prior to 1940 | 
# |**DIS**  | weighted distances to five Boston employment centres | 
# |**RAD**  | index of accessibility to radial highways | 
# |**TAX**  | full-value property-tax rate per $10,000 | 
# |**PTRATIO**  | pupil-teacher ratio by town | 
# |**B**  | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town | 
# |**LSTAT**  | % lower status of the population | 
# |**MEDV**  | Median value of owner-occupied homes in \$1000's | 

# In[ ]:


# create feature name columns
col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[ ]:


# redefine feature names
df.columns = col_name


# ## preprocessing

# In[ ]:


# verify there are no missing items in the dataset
df.isnull().sum()


# ## exploratory data anaysis (eda)

# In[ ]:


# short statistics on the data
df.describe()


# ## data visualization

# In[ ]:


# visualize features relationships
sns.pairplot(df, height=1.5);
plt.show()


# In[ ]:


# examine certain features
col_study = ['ZN', 'INDUS', 'NOX', 'RM']


# In[ ]:


sns.pairplot(df[col_study], height=2.5);
plt.show()


# | Code   | Description   |
# |:---|:---|
# |**CRIM** | per capita crime rate by town |
# |**ZN**  | proportion of residential land zoned for lots over 25,000 sq.ft. | 
# |**INDUS**  | proportion of non-retail business acres per town | 
# |**CHAS**  | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) | 
# |**NOX**  | nitric oxides concentration (parts per 10 million) | 
# |**RM**  | average number of rooms per dwelling | 
# |**AGE**  | proportion of owner-occupied units built prior to 1940 | 
# |**DIS**  | weighted distances to five Boston employment centres | 
# |**RAD**  | index of accessibility to radial highways | 
# |**TAX**  | full-value property-tax rate per $10,000 | 
# |**PTRATIO**  | pupil-teacher ratio by town | 
# |**B**  | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town | 
# |**LSTAT**  | % lower status of the population | 
# |**MEDV**  | Median value of owner-occupied homes in \$1000's | 

# In[ ]:


col_study = ['PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[ ]:


sns.pairplot(df[col_study], height=2.5);
plt.show()


# ## correlation analysis  

# In[ ]:


pd.options.display.float_format = '{:,.2f}'.format


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'MEDV']].corr(), annot=True)
plt.show()


# ## linear regression with scikit-learn

# In[ ]:


df.head()


# In[ ]:


X = df['RM'].values.reshape(-1,1)


# In[ ]:


y = df['MEDV'].values


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(X, y)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


plt.figure(figsize=(12,10));
sns.regplot(X, y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();


# In[ ]:


sns.jointplot(x='RM', y='MEDV', data=df, kind='reg', height=15, ratio=5, space=0.2, dropna=False, xlim=None, ylim=None,);
plt.show();


# ***

# In[ ]:


X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
model.fit(X, y)
plt.figure(figsize=(12,10));
sns.regplot(X, y);
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();


# In[ ]:


sns.jointplot(x='LSTAT', y='MEDV', data=df, kind='reg', height=12);
plt.show();


# ***

#  ## robust regression

# In[ ]:


df.head()


# ### RANdom SAmple Consensus (RANSAC) Algorithm
# 
# [http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression]
# (http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression)

# Each iteration performs the following steps:
# 
# 1. Select `min_samples` random samples from the original data and check whether the set of data is valid 
# (see `is_data_valid`).
# 
# 2. Fit a model to the random subset (`base_estimator.fit`) and check whether the estimated model is valid 
# (see `is_model_valid`).
# 
# 3. Classify all data as inliers or outliers by calculating the residuals to the estimated model 
# (`base_estimator.predict(X) - y`) - all data samples with absolute residuals smaller than the `residual_threshold` 
# are considered as inliers.
# 
# 4. Save fitted model as best model if number of inlier samples is maximal. In case the current estimated model 
# has the same number of inliers, it is only considered as the best model if it has better score.

# In[ ]:


X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values


# In[ ]:


from sklearn.linear_model import RANSACRegressor


# In[ ]:


ransac = RANSACRegressor()


# In[ ]:


ransac.fit(X, y)


# In[ ]:


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


# In[ ]:


np.arange(3, 10, 1)


# In[ ]:


line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))


# In[ ]:


sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,10));
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper left')
plt.show()


# In[ ]:


ransac.estimator_.coef_


# In[ ]:


ransac.estimator_.intercept_


# In[ ]:


X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(0, 40, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))


# In[ ]:


sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,10));
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper right')
plt.show()


# ***

# ## performance evaluation 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# X = df['LSTAT'].values.reshape(-1,1)
X = df.iloc[:, :-1].values


# In[ ]:


y = df['MEDV'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train, y_train)


# In[ ]:


y_train_pred = lr.predict(X_train)


# In[ ]:


y_test_pred = lr.predict(X_test)


# ### method: residual analysis

# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
plt.xlim([-10, 50])
plt.show()


# ***

# ### method: mean squared error (mse)
# 
# $$MSE=\frac{1}{n}\sum^n_{i=1}(y_i-\hat{y}_i)^2$$
# 
# * The average value of the Sums of Squared Error cost function  
# 
# * Useful for comparing different regression models 
# 
# * For tuning parameters via a grid search and cross-validation

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mean_squared_error(y_train, y_train_pred)


# In[ ]:


mean_squared_error(y_test, y_test_pred)


# ### method: coefficient of determination, $R^2$
# 
# $$R^2 = 1 - \frac{SSE}{SST}$$
# 
# SSE: Sum of squared errors
# 
# SST: Total sum of squares

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_train, y_train_pred)


# In[ ]:


r2_score(y_test, y_test_pred)


# ***

# ## final review

# ### performance of a good model

# In[ ]:


generate_random = np.random.RandomState(0)
x = 10 * generate_random.rand(1000)
y = 3 * x + np.random.randn(1000)
plt.figure(figsize = (10, 8))
plt.scatter(x, y);
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_train.reshape(-1, 1), y_train)


y_train_pred = model.predict(X_train.reshape(-1, 1))
y_test_pred = model.predict(X_test.reshape(-1, 1))


# ### method: residual analysis

# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-3, xmax=33, lw=2, color='k')
plt.xlim([-5, 35])
plt.ylim([-25, 15])
plt.show()


# ### method: mean squared error (mse)

# In[ ]:


mean_squared_error(y_train, y_train_pred)


# In[ ]:


mean_squared_error(y_test, y_test_pred)


# ### method: coefficient of determination, $R^2$

# In[ ]:


r2_score(y_train, y_train_pred)


# In[ ]:


r2_score(y_test, y_test_pred)


# References: Wikipedia, Anthony Ng
