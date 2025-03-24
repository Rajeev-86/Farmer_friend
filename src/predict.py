import cloudpickle as cp
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
    sarimax_input = pd.DataFrame(sarimax_input)
    # Load preprocessor and model
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Demand_Predictor', 'cp_preprocessor_SD.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cp.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Demand_Predictor', 'cp_model_SD.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cp.load(file)
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
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Soil-Climate_Compatibility_Classifier', 'cp_preprocessor_SC.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cp.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Soil-Climate_Compatibility_Classifier', 'cp_model_SC.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cp.load(file)
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
    PREPROCESS_PATH = os.path.join(BASE_DIR, 'models', 'Yield_Regression', 'cp_preprocessor_YR.pkl')
    with open(PREPROCESS_PATH, 'rb') as file:
        preprocessor = cp.load(file)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Yield_Regression', 'cp_model_YR.pkl')
    with open(MODEL_PATH, 'rb') as file:
        model = cp.load(file)
    # Transform and make predictions
    data_transformed = preprocessor.transform(yield_input)
    predicted_yield = model.predict(data_transformed)
    return predicted_yield
