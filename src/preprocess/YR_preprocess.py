import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def fit_preprocessing(X_train):
    """Fits OneHotEncoder and StandardScaler on training data."""
    
    # Using sin-cos transformation for 'Month' (makes transitions smooth)
    X_train = X_train.copy()
    X_train['Month_sin'] = np.sin(2 * np.pi * X_train['Month'] / 12)
    X_train['Month_cos'] = np.cos(2 * np.pi * X_train['Month'] / 12)
    X_train.drop(columns=['Month'], inplace=True)

    # Converting 'Year' to relative years since 2015
    X_train['Year'] = X_train['Year'] - 2015

    # Identify categorical & numerical columns
    cat_columns = ['Crop', 'Region']
    num_columns = X_train.drop(columns=cat_columns).columns

    # Initialize encoders
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    scaler = StandardScaler()

    # Fit encoders on training data
    ohe.fit(X_train[cat_columns])
    scaler.fit(X_train[num_columns])

    return ohe, scaler  # Return fitted encoders

def transform_preprocessing(X, ohe, scaler):
    """Applies fitted OneHotEncoder and StandardScaler to new data."""
    
    X = X.copy()

    # Apply same transformations as training set
    X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
    X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
    X.drop(columns=['Month'], inplace=True)

    X['Year'] = X['Year'] - 2015

    cat_columns = ['Crop', 'Region']
    num_columns = X.drop(columns=cat_columns).columns

    # Transform using fitted encoders
    cat_data = ohe.transform(X[cat_columns])
    num_data = scaler.transform(X[num_columns])

    # Convert categorical data to DataFrame
    cat_data = pd.DataFrame(cat_data, columns=ohe.get_feature_names_out(cat_columns))
    num_data = pd.DataFrame(num_data, columns=num_columns)

    # Reset index to avoid mismatches during concatenation
    X_transformed = pd.concat([num_data.reset_index(drop=True), cat_data.reset_index(drop=True)], axis=1)

    return X_transformed

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess_data(data):

  #Handling 'Date' column
  data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
  data.set_index('Date', inplace=True)
  data.drop(columns=['Year', 'Month'], inplace=True)
  data = data.reset_index()

  temp_data = data.copy()

  #Separate the features
  numeric_columns = temp_data.drop(columns=['Crop', 'Region', 'Date']).columns.tolist()
  categorical_columns = ['Crop', 'Region']
  data_to_transform = temp_data.drop(columns=['Date'])  # Exclude 'Date' column

  # Use ColumnTransformer to apply transformations to specific columns
  preprocessor = ColumnTransformer(
      transformers=[
          ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns),  # One-Hot Encoding for categorical variables
          ('num', StandardScaler(), numeric_columns)  # StandardScaler for numeric variables
      ],
      remainder='drop'  # This keeps the remaining columns (like Year and Month) intact
  )

  # Apply the transformations to the dataset
  processed_data = preprocessor.fit_transform(data)

  # Get the transformed column names for categorical columns (after one-hot encoding)
  encoded_columns = preprocessor.transformers_[0][1].get_feature_names_out(categorical_columns)

  # Combine the new column names with the original columns
  columns = list(encoded_columns) + numeric_columns

  temp_data = pd.DataFrame(processed_data, columns=columns)

  # Now you have the transformed data with Year and Month intact
  temp_data['Date'] = data['Date']
  data = temp_data

  return data

def preprocess_dtypes(data):
  data = data.select_dtypes(include=['number'])
  return data


import cloudpickle
import os
import sys
import pandas as pd
from typing import Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from src.preprocess.SC_preprocess import transform_preprocessor
from src.preprocess.SD_preprocess import preprocess_data, preprocess_dtypes
from src.preprocess.YR_preprocess import transform_preprocessing

custom_globals = {
    'preprocess_data': preprocess_data,
    'preprocess_dtypes': preprocess_dtypes,
    'transform_preprocessor': transform_preprocessor,
    'transform_preprocessing': transform_preprocessing
}

def predict_market_demand(data: dict[str, Any]):

    sarimax_input = {
        "year": data["year"],
        "month": data["month"],
        "crop": data["crop"],  # Ensure the crop is passed correctly
        "region": data["region"],
        "temperature": data["temperature"],
        "rainfall": data["rainfall"],
        "humidity": data["humidity"],
        "soil_pH": data["soil_pH"],
        "soil_nitrogen": data["soil_nitrogen"],
        "supply_tons": data["supply_tons"],
        "import_tons": data["import_tons"],
        "export_tons": data["export_tons"],
    }
    # Load preprocessor and model
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Demand_Predictor', 'preprocessor_SD.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cloudpickle.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Demand_Predictor', 'model_SD.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cloudpickle.load(file)
    # Transform and make predictions
    data_transformed = preprocessor.transform(sarimax_input)
    predicted_demand = model.predict(start=0, end=0, exog=data_transformed)
    return predicted_demand

def predict_compatibility(data: dict[str, Any]):

    classifier_input = {
        "crop_type": data["crop"],
        "farm_size_acres": data["farm_size_acres"],
        "irrigation_available": data["irrigation_available"],
        "soil_pH": data["soil_pH"],
        "soil_nitrogen": data["soil_nitrogen"],
        "soil_organic_matter": data["soil_organic_matter"],
        "temperature": data["temperature"],
        "rainfall": data["rainfall"],
        "humidity": data["humidity"],
    }
    # Load preprocessor and model
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Soil-Climate_Compatibility_Classifier', 'preprocessor_SC.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cloudpickle.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Soil-Climate_Compatibility_Classifier', 'model_SC.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cloudpickle.load(file)
    # Transform and make prediction
    data_transformed = preprocessor.transform(classifier_input)
    predicted_compatibility = model.predict(data_transformed)
    return predicted_compatibility

def predict_yield(data: dict[str, Any]):

    yield_input = {
        "year": data["year"],
        "month": data["month"],
        "crop": data["crop"],
        "region": data["region"],
        "temperature": data["temperature"],
        "rainfall": data["rainfall"],
        "humidity": data["humidity"],
        "soil_pH": data["soil_pH"],
        "soil_nitrogen": data["soil_nitrogen"],
        "soil_phosphorus": data["soil_phosphorus"],
        "soil_potassium": data["soil_potassium"],
        "fertilizer_use": data["fertilizer_use"],
        "pesticide_use": data["pesticide_use"],
        "previous_year_yield": data["previous_year_yield"],
        "sowing_to_harvest_days": data["sowing_to_harvest_days"],
    }
    # Load preprocessor and model
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Yield_Regression', 'preprocessor_YR.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cloudpickle.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Yield_Regression', 'YR_model.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cloudpickle.load(file)
    # Transform and make predictions
    data_transformed = preprocessor.transform(yield_input)
    predicted_yield = model.predict(data_transformed)
    return predicted_yield
