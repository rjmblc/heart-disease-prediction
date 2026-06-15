import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

# load .env contents to env
load_dotenv()


PROJECT_PATH = Path(os.getenv("PROJECT_PATH")).resolve()
MODEL_PATH = PROJECT_PATH / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_PATH / os.getenv("LOG_DIR") /os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents= True, exist_ok= True)

# define log class
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# load the trained model 
logging.info("Model loading started")
model = load(MODEL_PATH)
logging.info("Model loaded successfully")

def predict(input_data: dict):

    df = pd.DataFrame([input_data])

    # get predicted class
    prediction  = int(model.predict(df)[0])

    # get prediction probability
    probability = float(model.predict_proba(df)[0][1])

    logging.info(f"Model has a prediction of {prediction} with a probability of {probability}")

    return {
        "prediction" : prediction,
        "probability" : probability
    }