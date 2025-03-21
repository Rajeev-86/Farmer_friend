from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List
import numpy as np
import os

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define input schema
class CropRequest(BaseModel):
    year: int
    month: int
    region: str
    temperature: float
    rainfall: float
    humidity: float
    soil_pH: float
    soil_nitrogen: float
    soil_phosphorus: float
    soil_potassium: float
    fertilizer_use: float
    pesticide_use: float
    previous_year_yield: float
    sowing_to_harvest_days: int
    farm_size_acres: float
    irrigation_available: bool
    supply_tons: float
    import_tons: float
    export_tons: float
    crops: List[str]  # List of crops to evaluate

# Dummy model functions (Replace with actual model inference code)
def predict_market_demand(crop, data):
    return np.random.uniform(50, 500)  # Replace with SARIMAX model inference

def classify_soil_climate(crop, data):
    return np.random.choice([0, 1])  # Replace with actual classifier inference

def predict_yield(crop, data):
    return np.random.uniform(1, 10)  # Replace with yield regression model inference

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/recommend_crops")
def recommend_crops(request: CropRequest):
    scores = []
    w1, w2, w3 = 1, 1, 1  # Equal weights
    
    for crop in request.crops:
        market_demand = predict_market_demand(crop, request)
        compatibility = classify_soil_climate(crop, request)
        predicted_yield = predict_yield(crop, request)
        
        final_score = (w1 * market_demand) + (w2 * compatibility) + (w3 * predicted_yield)
        scores.append((crop, final_score))
    
    # Sort crops by final score in descending order
    ranked_crops = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return {"ranked_crops": ranked_crops}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7000)))
