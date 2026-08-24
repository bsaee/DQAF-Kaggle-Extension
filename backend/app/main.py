from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import json

from app.schemas.profile import ProfileRequest, ProfileResponse, ValidateKeyRequest
from app.services.kaggle_service import fetch_and_load_kaggle_dataset, set_kaggle_credentials
from app.engine.task_detector import detect_task_type
from app.engine.profiler import profile_dataset
from app.engine.stress_test import run_algorithmic_stress_test
from app.engine.remediator import generate_remediation_recipes

app = FastAPI(
    title="DQAF Backend API",
    description="Dataset Quality Assessment Framework API for Kaggle Datasets",
    version="1.0.0"
)

# Enable CORS for Chrome Extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from chrome-extension:// origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "framework": "DQAF v1.0.0"}

@app.post("/api/v1/validate-key")
def validate_kaggle_key(payload: ValidateKeyRequest):
    """Checks if the user's Kaggle credentials are valid."""
    try:
        set_kaggle_credentials(payload)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return {"valid": True, "message": "Kaggle credentials authenticated successfully."}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

@app.post("/api/v1/profile", response_model=ProfileResponse)
def generate_profile(payload: ProfileRequest):
    """
    Ingests Kaggle dataset, profiles data health, runs 3-model stress test,
    and returns full DQAF analytics.
    """
    try:
        df, file_name = fetch_and_load_kaggle_dataset(
            dataset_slug=payload.dataset_slug,
            auth=payload.auth,
            max_rows=payload.max_sample_rows
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest dataset: {str(e)}")

    # 1. Target & Task Detection
    task_info = detect_task_type(df, target_column=payload.target_column)
    target_col = task_info["target_column"]

    # 2. Statistical Profiling & Nutrition Vitals
    profile_data = profile_dataset(df, target_column=target_col)

    # 3. ML Degradation Stress Test (if classification task)
    stress_test_data = None
    if task_info["is_supported"] and target_col:
        try:
            stress_test_data = run_algorithmic_stress_test(
                df=df,
                target_col=target_col,
                max_sample_rows=payload.max_sample_rows
            )
        except Exception as e:
            stress_test_data = {"error": f"Stress test failed: {str(e)}"}

    # 4. Remediation Recipes
    remediation_recipes = []
    if target_col:
        remediation_recipes = generate_remediation_recipes(
            risk_flags=profile_data["risk_flags"],
            target_col=target_col,
            class_distribution=profile_data["class_distribution"]
        )

    return ProfileResponse(
        dataset_slug=payload.dataset_slug,
        target_detection=task_info,
        vitals=profile_data["vitals"],
        health_score=profile_data["health_score"],
        risk_flags=profile_data["risk_flags"],
        class_distribution=profile_data["class_distribution"],
        column_profiles=profile_data["column_profiles"],
        cramers_matrix=profile_data["cramers_matrix"],
        stress_test=stress_test_data,
        remediation_recipes=remediation_recipes
    )