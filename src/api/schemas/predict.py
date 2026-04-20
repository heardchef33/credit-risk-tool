from typing import Any, List, Optional

from pydantic import BaseModel
from src.processing.validation import LoanDefaultDataInputSchema

class PredictionResults(BaseModel): 
    errors: Optional[Any]
    version: str 
    predictions: Optional[List[float]]
    prediction_probabilities: Optional[List[List[float]]]

class MultipleLoanDefaultInputs(BaseModel): 
    inputs: List[LoanDefaultDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "int_rate" : 2,
                        "fico_range_high" : 6.0,
                        "inq_last_6mths" : 1.0,
                        "open_il_12m" : 1.0,
                        "acc_open_past_24mths" : 2.0,
                        "mort_acc" : 2,
                        "num_tl_op_past_12m" : 3.0,
                        "percent_bc_gt_75" : 4.0,
                        "sub_grade" : 'A1',
                        "term" : '36 months'
                    }
                ]
            }
        }