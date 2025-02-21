from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
from pydantic import BaseModel
import os
import pandas as pd

with open("farm_compatibility_model.pkl", "rb") as f:
    model = pickle.load(f)
print(model)

# Initialize FastAPI app
app = FastAPI()

origins = os.getenv("ALLOW_ORIGINS", "*")  # Default to "*"
origins = origins.split(",") if isinstance(origins, str) else origins
print("Allowed origins:", origins)  # Debugging step

# Convert to a list if necessary
if isinstance(origins, str):
    origins = origins.split(",")  # Converts "http://example.com,https://example.com" to a list

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

 #Define request body structure
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

@app.get("/")
def home():
    return {"message": "API is up and running!"}

@app.post("/predict/")
def predict_farm_compatibility(data: FarmInput):
    # Convert input to a list (ensure order matches model training)

    irrigation = 1 if data.Irrigation_Available else 0

    input_data = pd.DataFrame([[
    data.Crop_Type, data.Farm_Size_Acres, irrigation,
    data.Soil_pH, data.Soil_Nitrogen, data.Soil_Organic_Matter,
    data.Temperature, data.Rainfall, data.Humidity
]], columns=[
    "Crop_Type", "Farm_Size_Acres", "Irrigation_Available",
    "Soil_pH", "Soil_Nitrogen", "Soil_Organic_Matter",
    "Temperature", "Rainfall", "Humidity"
])

    print("Input DataFrame:\n", input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]
    return {"compatible": bool(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Replace 8000 with your desired port
