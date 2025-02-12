import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
with open("farm_compatibility_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define request body structure
class FarmInput(BaseModel):
    Crop_Type: str
    Farm_Size_Acres: float
    Irrigation_Available: bool
    Soil_pH: float
    Soil_Nitrogen: float
    Soil_Organic_Matter: float
    Temperature: float
    Rainfall: float
    Humidity: float

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is up and running!"}

@app.post("/predict/")
def predict_farm_compatibility(data: FarmInput):
    # Convert input to a list (ensure order matches model training)
    input_data = [[
        data.Crop_Type, data.Farm_Size_Acres, data.Irrigation_Available,
        data.Soil_pH, data.Soil_Nitrogen, data.Soil_Organic_Matter,
        data.Temperature, data.Rainfall, data.Humidity
    ]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return {"compatible": bool(prediction)}

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
