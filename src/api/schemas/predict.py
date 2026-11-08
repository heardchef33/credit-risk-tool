from typing import Any, List, Optional

from pydantic import BaseModel

# Request schemas live in src.processing.validation so the API and the model
# package validate inputs with exactly the same rules.
from src.processing.validation import LoanDefaultDataInputSchema, MultipleLoanDefaultInputs

__all__ = ["LoanDefaultDataInputSchema", "MultipleLoanDefaultInputs", "PredictionResults"]


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]
    prediction_probabilities: Optional[List[List[float]]]
