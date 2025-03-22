import gdown
import os

MODEL_URL = "https://drive.google.com/file/d/1rnoQhjp59h89j6vK2YjcqBYXTNwMc0jf/view?usp=drive_link"
MODEL_PATH = "models/Demand_Predictor/model_SD.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading large model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model()
