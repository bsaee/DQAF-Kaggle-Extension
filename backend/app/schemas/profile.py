from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class KaggleAuth(BaseModel):
    username: str
    key: str

class ProfileRequest(BaseModel):
    dataset_slug: str = Field(..., description="Kaggle dataset slug in 'owner/dataset' format")
    target_column: Optional[str] = Field(None, description="User selected target column override")
    task_mode: Optional[str] = Field("auto", description="User motive: 'auto', 'classification', 'regression', 'unsupervised'")
    auth: Optional[KaggleAuth] = Field(None, description="BYOK Kaggle credentials")
    max_sample_rows: int = Field(5000, ge=500, le=20000)

class ValidateKeyRequest(BaseModel):
    username: str
    key: str

class ProfileResponse(BaseModel):
    dataset_slug: str
    target_detection: Dict[str, Any]
    vitals: Dict[str, Any]
    health_score: int
    risk_flags: Dict[str, Any]
    class_distribution: Dict[str, float]
    column_profiles: List[Dict[str, Any]]
    cramers_matrix: Dict[str, Dict[str, float]]
    stress_test: Optional[Dict[str, Any]] = None
    remediation_recipes: List[Dict[str, str]]