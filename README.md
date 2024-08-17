
# Bank Marketing Data Classification

This project demonstrates the process of preparing and modeling a bank marketing dataset using various resampling techniques and a RandomForestClassifier. The goal is to predict whether a client will subscribe to a term deposit (`y = yes/no`) based on various features.

## Table of Contents

- [Prepare the Data](#prepare-the-data)
- [RandomForestClassifier](#randomforestclassifier)
- [Random Undersampler](#random-undersampler)
- [Random Oversampler](#random-oversampler)
- [Cluster Centroids](#cluster-centroids)
- [SMOTE](#smote)
- [SMOTEENN](#smoteenn)
- [Results](#results)

## Prepare the Data

```python
# Import modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a Pandas DataFrame
bank_data_df = pd.read_csv('../Resources/bank.csv')

# Review the DataFrame
bank_data_df.head()

# Split the features and target data
y = bank_data_df['y']
X = bank_data_df.drop(columns='y')

# Encode the features dataset's categorical variables using get_dummies
X = pd.get_dummies(X)

# Review the features DataFrame
X.head()

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Instantiate a StandardScaler instance
scaler = StandardScaler()

# Fit the training data to the standard scaler
X_scaler = scaler.fit(X_train)

# Transform the training data using the scaler
X_train_scaled = X_scaler.transform(X_train)

# Transform the testing data using the scaler
X_test_scaled = X_scaler.transform(X_test)
```

## RandomForestClassifier

```python
# Import the RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier

# Instantiate a RandomForestClassifier instance
model = RandomForestClassifier()

# Fit the training data to the model
model.fit(X_train_scaled, y_train)

# Predict labels for original scaled testing features
y_pred = model.predict(X_test_scaled)
```

## Random Undersampler

```python
# Import RandomUnderSampler from imblearn
from imblearn.under_sampling import RandomUnderSampler

# Instantiate a RandomUnderSampler instance
rus = RandomUnderSampler(random_state=1)

# Fit the training data to the random undersampler model
X_undersampled, y_undersampled = rus.fit_resample(X_train_scaled, y_train)

# Count distinct values for the resampled target data
y_undersampled.value_counts()

# Instantiate a new RandomForestClassier model
model_undersampled = RandomForestClassifier()

# Fit the undersampled data to the new model
model_undersampled.fit(X_undersampled, y_undersampled)

# Predict labels for undersampled testing features
y_pred_undersampled = model_undersampled.predict(X_test_scaled)
```

## Random Oversampler

```python
# Import RandomOverSampler from imblearn
from imblearn.over_sampling import RandomOverSampler

# Instantiate a RandomOversampler instance
ros = RandomOverSampler(random_state=1)

# Fit the training data to the `RandomOverSampler` model
X_oversampled, y_oversampled = ros.fit_resample(X_train_scaled, y_train)

# Count distinct values for the resampled target data
y_oversampled.value_counts()

# Instantiate a new RandomForestClassier model
model_oversampled = RandomForestClassifier()

# Fit the oversampled data to the new model
model_oversampled.fit(X_oversampled, y_oversampled)

# Predict labels for oversampled testing features
y_pred_oversampled = model_oversampled.predict(X_test_scaled)
```

## Cluster Centroids

```python
# Import ClusterCentroids from imblearn
from imblearn.under_sampling import ClusterCentroids

# Instantiate a ClusterCentroids instance
cc_sampler = ClusterCentroids(random_state=1)

# Fit the training data to the cluster centroids model
X_resampled, y_resampled = cc_sampler.fit_resample(X_train_scaled, y_train)

# Count distinct values for the resampled target data
y_resampled.value_counts()

# Instantiate a new RandomForestClassier model
cc_model = RandomForestClassifier()

# Fit the resampled data to the new model
cc_model.fit(X_resampled, y_resampled)

# Predict labels for resampled testing features
cc_y_pred = cc_model.predict(X_test_scaled)
```

## SMOTE

```python
# Import SMOTE from imblearn
from imblearn.over_sampling import SMOTE

# Instantiate the SMOTE instance 
smote_sampler = SMOTE(random_state=1, sampling_strategy='auto')

# Fit the training data to the smote_sampler model
X_resampled, y_resampled = smote_sampler.fit_resample(X_train_scaled, y_train)

# Count distinct values for the resampled target data
y_resampled.value_counts()

# Instantiate a new RandomForestClassier model 
smote_model = RandomForestClassifier()

# Fit the resampled data to the new model
smote_model.fit(X_resampled, y_resampled)

# Predict labels for resampled testing features
smote_y_pred = smote_model.predict(X_test_scaled)
```

## SMOTEENN

```python
# Import SMOTEENN from imblearn
from imblearn.combine import SMOTEENN

# Instantiate the SMOTEENN instance
smote_enn = SMOTEENN(random_state=1)

# Fit the model to the training data
X_resampled, y_resampled = smote_enn.fit_resample(X_train_scaled, y_train)

# Instantiate a new RandomForestClassier model
smoteenn_model = RandomForestClassifier()

# Fit the resampled data to the new model
smoteenn_model.fit(X_resampled, y_resampled)

# Predict labels for resampled testing features
smoteenn_y_pred = smoteenn_model.predict(X_test_scaled)
```

## Results

### Classification Report - Original Data

```plaintext
              precision    recall  f1-score   support

          no       0.89      0.98      0.94       988
         yes       0.60      0.20      0.29       143

    accuracy                           0.88      1131
   macro avg       0.74      0.59      0.62      1131
weighted avg       0.86      0.88      0.85      1131
```

### Classification Report - Undersampled Data

```plaintext
              precision    recall  f1-score   support

          no       0.96      0.81      0.88       988
         yes       0.38      0.79      0.51       143

    accuracy                           0.81      1131
   macro avg       0.67      0.80      0.70      1131
weighted avg       0.89      0.81      0.84      1131
```

### Classification Report - Oversampled Data

```plaintext
              precision    recall  f1-score   support

          no       0.91      0.97      0.94       988
         yes       0.62      0.30      0.41       143

    accuracy                           0.89      1131
   macro avg       0.76      0.64      0.67      1131
weighted avg       0.87      0.89      0.87      1131
```

### Classification Report - Resampled Data - CentroidClusters

```plaintext
              precision    recall  f1-score   support

          no       0.99      0.29      0.45       988
         yes       0.17      0.99      0.29       143

    accuracy                           0.38      1131
   macro avg       0.58      0.64      0.37      1131
weighted avg       0.89      0.38      0.43      1131
```

### Classification Report - Resampled Data - SMOTE

```plaintext
              precision    recall  f1-score   support

          no       0.90      0.96      0.93       988
         yes       0.51      0.29      0.37       143

    accuracy                           0.88      1131
   macro avg       0.71      0.63      0.65      1131
weighted avg       0.85      0.88      0.86      1131
```

### Classification Report - Resampled Data - SMOTEENN

```plaintext
              precision    recall  f1-score   support

          no       0.93      0.91      0.92       988
         yes       0.46      0.55      0.50       143

    accuracy                           0.86      1131
   macro avg       0.69      0.73      0.71      1131
weighted avg       0.87      0.86      0.87      1131


```

## Conclusion

The project demonstrates the application of different resampling techniques (undersampling, oversampling, and SMOTE variants) combined with a RandomForestClassifier to tackle the issue of class imbalance in the bank marketing dataset. The results show varying levels of precision, recall, and f1-score for the positive class (`yes`), highlighting the challenges of imbalanced datasets and the impact of different resampling strategies.
```

This README should help document the process and results of your project. You can further customize it to match your project's specific needs and audience.
