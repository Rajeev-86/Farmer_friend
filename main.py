from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
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

# Dummy model functions (Replace with actual model inference code)
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
    return np.random.uniform(50, 500)  # Replace with SARIMAX model inference

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

    return np.random.choice([0, 1])  # Replace with actual classifier inference

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
    return np.random.uniform(1, 10)  # Replace with yield regression model inference

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

        market_demand = predict_market_demand(crop_data)
        compatibility = predict_compatibility(crop_data)
        predicted_yield = predict_yield(crop_data)

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
