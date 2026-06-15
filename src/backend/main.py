from fastapi import FastAPI
from pydantic import BaseModel

from src.backend.predictor import predict

app = FastAPI(
    title= "Heart Disease Prediction Application",
    version= "1.0"
)

# input schema matching training
class HeartDiseaseTraining(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# health check status
@app.get("/health")
def health_check():
    return {
        "status" : "ok"
    } 

# heart disease prediction endpoint
@app.post("/predict")
def predict_heart_disease(input_data : HeartDiseaseTraining):
    input_data = input_data.model_dump()
    result = predict(input_data = input_data)
    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "diagnosis": (
            "Heart Disease Detected"
            if result["prediction"] == 1
            else "No Heart Disease Detected"
        )
    }