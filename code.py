import pandas as pd
import numpy as np

dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
numeric_features = dataset.select_dtypes(include=['int64', 'float64']).drop(['RowNumber','CustomerId','Exited'],axis=1).columns



'''
from sklearn.compose import ColumnTransformer
scaled_check = ColumnTransformer(
    transformers=[
         ('num ', numeric_transformer,numeric_features)
    ])
scaled_columns = encoded_check.fit_transform(dataset)
scaled_columns[0:5]
'''



categorical_transformer = Pipeline(steps=[
   ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
categorical_features = dataset.select_dtypes(include=['object']).drop(['Surname'],axis=1).columns



'''
encoded_check = ColumnTransformer(
    transformers=[
         ('cat', categorical_transformer, categorical_features)
    ])
encoded_columns = encoded_check.fit_transform(dataset)
encoded_columns[0:5]
'''


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
      ])
      
      
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8,6,1), max_iter=300,activation = 'tanh',solver='adam',random_state=123)
pipe = Pipeline(steps=[('pre',preprocessor),('mlpc', mlp)])
pipe.fit(X_train,y_train) 

pipe.score(X_test,y_test)
