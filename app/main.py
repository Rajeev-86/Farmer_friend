from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from src.predict import predict_market_demand, predict_compatibility, predict_yield

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
    soil_organic_matter: float
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

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/recommend_crops")
def recommend_crops(request: CropRequest):
    
    weights = {"market_demand": 1, "compatibility": 1, "predicted_yield": 1}    
    crop_scores = []
    
    for crop in request.crops:
        # Convert request object to dictionary
        crop_data = request.model_dump()
        crop_data["crop"] = crop  # Add current crop name

        market_demand = float(predict_market_demand(crop_data))
        compatibility = float(predict_compatibility(crop_data))
        predicted_yield = float(predict_yield(crop_data))

        # Compute final score
        final_score = (
            weights["market_demand"] * market_demand +
            weights["compatibility"] * compatibility +
            weights["predicted_yield"] * predicted_yield
        )

        crop_scores.append({"crop": crop, "score": final_score})

    # Sort crops by score in descending order
    ranked_crops = sorted(crop_scores, key=lambda x: x["score"], reverse=True)

    return {"ranked_crops": ranked_crops}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7000)))
