import numpy as np
import glob
import pandas as pd
from sklearn import preprocessing

encoded_df = pd.read_csv('trial.csv')

encoded_df.drop(encoded_df.columns[0], inplace=True, axis=1)

from datetime import datetime
encoded_df['start_date'] = encoded_df['start_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

X_train = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['start_date'].dt.year <= 2016]['total_runs'].values
y_test = encoded_df[encoded_df['start_date'].dt.year >= 2017]['total_runs'].values

           # Removing the 'date' column

X_train.drop(labels='start_date', axis=True, inplace=True)
X_test.drop(labels='start_date', axis=True, inplace=True)

           # --- Model Building ---
           # Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

import joblib
joblib.dump(regressor, 'model.joblib')



