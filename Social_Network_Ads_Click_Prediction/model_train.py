import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('Social_Network_Ads.csv')

print(df)

if 'User ID' in df.columns:
  df.drop(columns = 'User ID', inplace=True)

X = df.drop('Purchased', axis=1)
y = df['Purchased']

numeric_features = X.select_dtypes(include = ['int64', 'float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

num_features_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
)

cat_feature_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OrdinalEncoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_features_transformer, numeric_features),
        ('cat', cat_feature_transformer, categorical_features)
    ]
)

svm_model = SVC(C=7, gamma='scale', kernel='poly', random_state=42)

svm_model_pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('svm_model', svm_model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

svm_model_pipeline.fit(X_train, y_train)

y_pred = svm_model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model_pipeline, f)

print("SVM model pipeline saved as svm_model.pkl")