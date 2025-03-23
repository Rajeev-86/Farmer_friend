def predict_market_demand(data: Dict):

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
    # Load the saved SARIMAX model and preprocessor
    with open('/content/drive/MyDrive/Colab Notebooks/crops recommender/supply demand model/model_SD.pkl', 'rb') as model_file:
        model_fit = pickle.load(model_file)

    with open('/content/drive/MyDrive/Colab Notebooks/crops recommender/supply demand model/preprocessor_SD.pkl', 'rb') as prep_file:
        preprocessor = pickle.load(prep_file)

    # Transform input features
    new_data_transformed = preprocessor.transform(new_data)

    # Predict Market Demand (returns scaled values)
    predicted_demand = model_fit.predict(start=0, end=0, exog=new_data_transformed)

    return predicted_demand

def predict_compatibility(data: Dict):

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

    with open('/content/drive/MyDrive/Colab Notebooks/crops recommender/Soil-Climate model/preprocessor_SC.pkl', 'rb') as prep_file:
        preprocessor = pickle.load(prep_file)
    with open('/content/drive/MyDrive/Colab Notebooks/crops recommender/Soil-Climate model/model_SC.pkl', 'rb') as model_file:
        model_fit = pickle.load(model_file)

    # Transform input features
    new_data_transformed = preprocessor.transform(new_data)

    # Predict Market Demand (returns scaled values)
    predicted_compatibility = model_fit.predict(new_data_transformed)

    return predicted_compatibility

def predict_yield(data: Dict):

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
    with open('/content/drive/MyDrive/Colab Notebooks/crops recommender/Yield Regression Model/preprocessor_YR.pkl', 'rb') as prep_file:
        preprocessor = pickle.load(prep_file)
    with open('/content/drive/MyDrive/Colab Notebooks/crops recommender/Yield Regression Model/YR_model.pkl', 'rb') as model_file:
        model_fit = pickle.load(model_file)

    # Transform input features
    new_data_transformed = preprocessor.transform(new_data)

    # Predict Market Demand (returns scaled values)
    predicted_yield = model_fit.predict(new_data_transformed)

    return predicted_yield
