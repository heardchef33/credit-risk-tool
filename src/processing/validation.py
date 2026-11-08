import json
from typing import Annotated, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.config.core import config

# Allowed category values. These must match the categories the trained
# encoders saw during fitting: anything else makes OneHotEncoder /
# OrdinalEncoder raise "Found unknown categories" at predict time.
TERMS = ("36 months", "60 months")
SUB_GRADES = tuple(f"{grade}{level}" for grade in "ABCDEFG" for level in range(1, 6))

Term = Literal["36 months", "60 months"]
SubGrade = Literal[SUB_GRADES]  # type: ignore[valid-type]

# Numeric fields are Optional (nullable) because the pipeline imputes missing
# values with the training median, but every field must still be *present*
# in the request so that typos in field names are caught instead of being
# silently imputed.
NonNegativeCount = Annotated[Optional[float], Field(ge=0, le=100)]
Percentage = Annotated[Optional[float], Field(ge=0, le=100)]


class LoanDefaultDataInputSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "int_rate": 10.5,
                "fico_range_high": 750.0,
                "inq_last_6mths": 1.0,
                "open_il_12m": 2.0,
                "acc_open_past_24mths": 3.0,
                "mort_acc": 1.0,
                "num_tl_op_past_12m": 2.0,
                "percent_bc_gt_75": 25.0,
                "sub_grade": "A2",
                "term": "36 months",
            }
        },
    )

    int_rate: Annotated[Optional[float], Field(ge=0, le=50, description="Interest rate (%)")]
    fico_range_high: Annotated[Optional[float], Field(ge=300, le=850, description="Upper bound of FICO range")]
    inq_last_6mths: NonNegativeCount
    open_il_12m: NonNegativeCount
    acc_open_past_24mths: NonNegativeCount
    mort_acc: NonNegativeCount
    num_tl_op_past_12m: NonNegativeCount
    percent_bc_gt_75: Percentage
    term: Optional[Term]
    sub_grade: Optional[SubGrade]

    @field_validator("term", "sub_grade", mode="before")
    @classmethod
    def _normalise_category(cls, value):
        # The raw Lending Club data has values like " 36 months"; accept
        # surrounding whitespace and lower-case grades instead of rejecting them.
        if isinstance(value, str):
            value = value.strip()
            return value.upper() if len(value) == 2 else value
        return value


class MultipleLoanDefaultInputs(BaseModel):
    inputs: Annotated[List[LoanDefaultDataInputSchema], Field(min_length=1, max_length=1000)]


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[list]]:
    """Validate incoming data and prepare it for the model pipeline.

    Returns the validated features and ``None``, or the original data and a
    JSON-serialisable list of validation errors.
    """
    features = config.model_settings.features

    missing = [col for col in features if col not in input_data.columns]
    if missing:
        return input_data, [
            {"type": "missing", "loc": [col], "msg": "Field required"} for col in missing
        ]

    records = input_data[features].replace({np.nan: None}).to_dict(orient="records")

    try:
        validated = MultipleLoanDefaultInputs(inputs=records)
    except ValidationError as error:
        return input_data, json.loads(error.json(include_url=False))

    validated_data = pd.DataFrame([row.model_dump() for row in validated.inputs])[features]

    # The imputers only treat np.nan as missing, not None, so normalise nulls.
    numerical = config.model_settings.numerical_features
    validated_data[numerical] = validated_data[numerical].astype(float)
    categorical = config.model_settings.categorical_features
    validated_data[categorical] = (
        validated_data[categorical].astype(object).where(validated_data[categorical].notna(), np.nan)
    )

    return validated_data, None
