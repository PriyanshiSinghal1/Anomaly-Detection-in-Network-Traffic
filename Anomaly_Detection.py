# importing libraries 

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('/content/test_Anamoly.csv')
df2 = pd.read_csv('/content/train_Anamoly.csv')

df1.head()
df1.tail()

df2.head()
df2.tail()

print(df1.isnull().sum())

print(df2.isnull().sum())

"""##NORMALIZE THE DATA"""

scaler = StandardScaler()
data_normalized = scaler.fit_transform(df2)

corr=df2.corr()

y_train = df2["is_anomaly"]
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

features = ["value", "predicted"]
x_train = df2[features]

plt.figure(figsize=(10, 6))
plt.scatter(x_train["value"][y_train == 0], x_train["predicted"][y_train == 0], c='blue', marker='s', label="Normal")
plt.scatter(x_train["value"][y_train == 1], x_train["predicted"][y_train == 1], c='red', marker='x', label="Anomaly")
plt.title("Training Data: Normal vs Anomaly")
plt.xlabel("Value")
plt.ylabel("Predicted")
plt.legend()
plt.grid(True)
plt.show()

random_forest = RandomForestClassifier(n_jobs=-1)
random_forest.fit(x_train, y_train)

x_test = df1[features]
predictions = random_forest.predict(x_test)

submission = pd.read_csv('/content/Submission_Anamoly.csv')

inv_predictions = label_encoder.inverse_transform(predictions)

results = pd.DataFrame({
    "timestamp": df1["timestamp"],
    "is_anomaly": inv_predictions
})
results

results.to_csv("Submission_Anamoly.csv", index=False)
