import pandas as pd 
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



breast_cancer = load_breast_cancer()
df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df['target'] = breast_cancer.target 
df['target'] = df['target'].map({0: 'malignant', 1: 'benign', })

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



# Compute mean of all features from training data
feature_means = X_train.mean()

# Ask for the 4 main features
feature1 = float(input("mean radius: "))
feature2 = float(input("mean texture: "))
feature3 = float(input("mean perimeter: "))
feature4 = float(input("mean area: "))

# Start with all features = mean
sample = feature_means.copy()

# Replace the 4 features you input
sample[['mean radius', 'mean texture', 'mean perimeter', 'mean area']] = [feature1, feature2, feature3, feature4]

# Convert to numpy array and reshape
sample = sample.values.reshape(1, -1)

# Predict
prediction = model.predict(sample)
proba = model.predict_proba(sample)

print("Predicted class:", prediction[0])
print("Probability of Benign:", round(proba[0][0], 2))
print("Probability of Malignant:", round(proba[0][1], 2))
