from typing import List, Optional, Tuple
import numpy as np
import pandas as pd 
from pydantic import BaseModel, ValidationError

from src.config.core import config 

class LoanDefaultDataInputSchema(BaseModel): 
    int_rate: Optional[float]
    fico_range_high: Optional[float]
    inq_last_6mths: Optional[float]
    open_il_12m: Optional[float]
    acc_open_past_24mths: Optional[float]
    mort_acc: Optional[float]
    num_tl_op_past_12m: Optional[float]
    percent_bc_gt_75: Optional[float]
    term: Optional[str]
    sub_grade: Optional[str]

class MultipleLoanDefaultInputs(BaseModel): 
    inputs: List[LoanDefaultDataInputSchema]

def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]: 
    """Validate features of incoming data."""
    # remove whitespaces 
    input_data['term'] = input_data['term'].str.strip()
    validated_data = input_data[config.model_settings.features].copy()
    errors = None

    try: 
        MultipleLoanDefaultInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error: 
        errors = error.json()
    
    return validated_data, errors






