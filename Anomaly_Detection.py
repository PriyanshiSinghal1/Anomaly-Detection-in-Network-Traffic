import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the datasets
train_df = pd.read_csv('/content/train (1).csv')
test_df = pd.read_csv('/content/test.csv')

# Display the first few rows of the train dataset
print(train_df.head())

# Preprocess the data
# We will use the 'value' column to train the model
X_train = train_df[['value']].values
X_test = test_df[['value']].values

# Initialize the model
model = IsolationForest(contamination=0.01)

# Fit the model on the training data
model.fit(X_train)

# Predict anomalies on the test data
test_df['is_anomaly'] = model.predict(X_test)

# Convert predictions: -1 (anomaly) to 1, and 1 (normal) to 0
test_df['is_anomaly'] = test_df['is_anomaly'].map({-1: 1, 1: 0})

# Prepare the submission file
submission_df = test_df[['timestamp', 'is_anomaly']]
submission_df.to_csv('/content/Submission (1).csv', index=False)

# Evaluate the model
# We need the ground truth values to calculate the F1 score
# Assuming 'is_anomaly' in the test data is the ground truth
y_true = test_df['is_anomaly'].values
y_pred = model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")

# Display the first few rows of the submission file
print(submission_df.head())

# Display the entire submission file for verification (optional)
print(submission_df)
