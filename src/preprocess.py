import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Target
    y = df['Price']
    X = df.drop('Price', axis=1)
    
    # Numeric features
    numeric_features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features
    categorical_features = ['Type', 'Method', 'Suburb']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return X, y, preprocessor
