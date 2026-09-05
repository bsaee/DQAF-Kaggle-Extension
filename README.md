# Kaggle Dataset ML Auditor & Nutrition Label

An on-platform data quality auditor and stress-testing extension for Kaggle datasets. The tool injects real-time dataset health metrics, simulates algorithmic degradation under data missingness, and generates production-ready Scikit-learn remediation code directly inside the browser.

## Architecture Overview

- **Frontend:** Manifest V3 Chrome Extension (`vanilla JS`, `Chart.js`) providing an in-page popup overlay and an expanded full-page analysis dashboard.

- **Backend:** `FastAPI` (Python) statistical engine leveraging `Scikit-learn`, `pandas`, and `NumPy` for feature diagnostics and machine learning stress-testing.

- **Data Layer:** Direct integration with the Kaggle API using a Bring-Your-Own-Key (BYOK) credential model.

## Core Capabilities

- **Task-Aware Diagnostics:** Automatically identifies or allows user selection of the ML objective:
  - **Classification:** Evaluates $F_1$-score degradation, class imbalance ratios, and Shannon Entropy ($H(X)$).
  - **Regression:** Evaluates $R^2$ degradation curves, target skewness ($\gamma_1$), kurtosis, and target distribution histograms.
  - **Unsupervised:** Evaluates PCA score variance capture curves and feature scale disparities.

- **Predictive Missingness Stress-Testing:** Simulates progressive data corruption (0% → 50% induced missingness) across linear models, random forests, and gradient boosting trees to quantify performance degradation.

- **Automated Data Hygiene:**
  - Identifies and excludes arbitrary index/UUID columns using regex pattern matching and uniqueness thresholds ($\geq 95\%$).
  - Detects pseudo-numeric categorical variables (e.g., discrete binary flags or low-cardinality ordinal codes) to prevent inappropriate continuous scaling.

- **Multivariate Leakage & Redundancy:**
  - Target leakage warnings ($r \geq 0.98$).
  - Pearson correlation matrices ($r \geq 0.85$) and Cramér's V associations for categorical features.

- **Automated Remediation:** Generates copyable, tailored Scikit-learn pipelines (e.g., `PowerTransformer`, `TargetEncoder`, balanced class weightings, and feature droppers).

## Project Structure

```plaintext
.
├── backend/
│   ├── app/
│   │   ├── engine/
│   │   │   ├── profiler.py          # Statistical vitals, entropy, skewness, & associations
│   │   │   ├── stress_test.py       # Cross-validation & missingness degradation suite
│   │   │   ├── task_detector.py     # Task inference & ID column pruning
│   │   │   └── remediator.py        # Scikit-learn code generation
│   │   ├── schemas/                 # Pydantic request/response contracts
│   │   ├── services/                # Kaggle API ingestion
│   │   └── main.py                  # FastAPI routing & middleware
│   └── requirements.txt
└── extension/
    ├── manifest.json                # Chrome Manifest V3 configuration
    ├── popup/                       # In-page nutrition label card
    └── dashboard/                   # Full analysis studio & Chart.js visualizations
```

## Getting Started

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Chrome Extension Setup

1. Open Google Chrome and navigate to `chrome://extensions/`.

2. Toggle on **Developer mode** in the top-right corner.

3. Click **Load unpacked** and select the `extension/` directory.

4. Open any Kaggle dataset page (e.g.,
   `https://www.kaggle.com/datasets/heptapod/titanic`).

5. Click the extension icon, add your Kaggle API credentials (BYOK), and run the audit.
