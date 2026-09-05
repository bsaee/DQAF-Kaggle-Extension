from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.profile import ProfileRequest, ProfileResponse, ValidateKeyRequest
from app.services.kaggle_service import fetch_and_load_kaggle_dataset, set_kaggle_credentials
from app.engine.task_detector import detect_task_type
from app.engine.profiler import profile_dataset
from app.engine.stress_test import run_task_benchmark
from app.engine.remediator import generate_remediation_recipes

app = FastAPI(
    title="DQAF Backend API",
    description="Dataset Quality Assessment Framework API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "framework": "DQAF v1.0.0"}

@app.post("/api/v1/validate-key")
def validate_kaggle_key(payload: ValidateKeyRequest):
    try:
        set_kaggle_credentials(payload)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return {"valid": True, "message": "Kaggle authenticated successfully."}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

@app.post("/api/v1/profile")
def generate_profile(payload: ProfileRequest):
    try:
        df, file_name = fetch_and_load_kaggle_dataset(
            dataset_slug=payload.dataset_slug,
            auth=payload.auth,
            max_rows=payload.max_sample_rows
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest dataset: {str(e)}")

    user_mode = (payload.task_mode or "auto").lower()

    # Task detection
    task_info = detect_task_type(df, target_column=payload.target_column, task_mode=user_mode)
    
    if user_mode in ["classification", "regression", "unsupervised"]:
        effective_task = user_mode
    else:
        effective_task = "regression" if "regression" in task_info["task_type"] else ("unsupervised" if "unsupervised" in task_info["task_type"] else "classification")

    target_col = None if effective_task == "unsupervised" else (payload.target_column or task_info["target_column"])
    
    if effective_task in ["classification", "regression"] and not target_col:
        target_col = df.columns[-1]
        task_info["target_column"] = target_col

    # Profiling
    profile_data = profile_dataset(df, target_column=target_col, task_type=effective_task)

    # Benchmarks
    benchmark_data = None
    try:
        benchmark_data = run_task_benchmark(
            df=df,
            target_col=target_col,
            task_type=effective_task,
            max_sample_rows=payload.max_sample_rows
        )
    except Exception as e:
        benchmark_data = {"error": f"Benchmark failed: {str(e)}"}

    # Remediations
    remediation_recipes = generate_remediation_recipes(
        risk_flags=profile_data["risk_flags"],
        target_col=target_col,
        task_mode=effective_task,
        regression_stats=profile_data.get("regression_stats")
    )

    return {
        "dataset_slug": payload.dataset_slug,
        "target_detection": task_info,
        "vitals": profile_data["vitals"],
        "health_score": profile_data["health_score"],
        "risk_flags": profile_data["risk_flags"],
        "class_distribution": profile_data["class_distribution"],
        "regression_stats": profile_data.get("regression_stats", {}),
        "target_histogram": profile_data.get("target_histogram"),
        "column_profiles": profile_data["column_profiles"],
        "cramers_matrix": profile_data["cramers_matrix"],
        "stress_test": benchmark_data,
        "remediation_recipes": remediation_recipes
    }