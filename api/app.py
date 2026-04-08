from fastapi import FastAPI
from api.models.sentiment import PredictRequest, PredictResponse
from api.services.inference import Inference


app = FastAPI()
inference = Inference("models")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    prediction = inference.predict(request.text)
    return PredictResponse(prediction=prediction)
