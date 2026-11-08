from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from src import __version__ as model_version
from src.api import __version__, schemas
from src.api.config import settings
from src.predict import make_prediction

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Health check, also used by the keep-alive pinger."""
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.model_dump()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleLoanDefaultInputs) -> Any:
    """
    Make loan status predictions using the XGBoost model.

    Invalid requests are rejected with 422 by the request schema before they
    reach the model.
    """

    input_df = pd.DataFrame([row.model_dump() for row in input_data.inputs])

    logger.info(f"Making prediction on {len(input_df)} input(s)")
    results = make_prediction(input_data=input_df)

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results['errors']}")
        raise HTTPException(status_code=400, detail=results["errors"])

    logger.info(f"Prediction results: {results['predictions']}")

    return {
        "errors": None,
        "version": results["version"],
        "predictions": results["predictions"].tolist(),
        "prediction_probabilities": results["prediction_probabilities"].tolist(),
    }
