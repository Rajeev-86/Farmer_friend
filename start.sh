uvicorn app.main:app --host 0.0.0.0 --port 10000

import gdown

url = "https://drive.google.com/file/d/1-0VpADZlJocxblOEZdiEwcz5tpBnNDyU/view?usp=drive_link"
output = "model_SD.pkl"
gdown.download(url, output, quiet=False)
