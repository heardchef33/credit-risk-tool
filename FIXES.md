# Fix Log

A record of the issues found in a full review of this repository (November 2026), what caused each one, how it was fixed and how the fix was verified.

| # | Issue | Severity |
|---|-------|----------|
| 1 | Predictions failed outside the importing thread | High |
| 2 | Invalid input crashed the API with 500 | High |
| 3 | Training script could not run | High |
| 4 | "Default probability" was not a probability | High |
| 5 | Tests could not catch bugs | Medium |
| 6 | Bloated runtime dependencies | Medium |
| 7 | Broken local Docker setup | Medium |
| 8 | Demo slow or failing after idle periods | Medium |
| 9 | Debug prints and Pydantic v1 leftovers | Low |

---

## 1. Predictions failed outside the importing thread

**Symptom.** A valid request to `/api/v1/predict` returned 500 under FastAPI's `TestClient`, with `ValueError: Specifying the columns using strings is only supported for dataframes.`

**Cause.** The pipeline's `ColumnTransformer`s select columns by name, so every step must output DataFrames. This was configured with `sklearn.set_config(transform_output="pandas")`, which is **thread-local**: it only affects the thread that called it. Production worked only because uvicorn runs the `async` endpoint on the importing thread. A sync endpoint, a threaded server or a test client received numpy arrays and crashed.

**Fix.** `_final_pipeline.set_output(transform="pandas")` right after loading the model (`src/predict.py`), so the setting belongs to the model rather than to a thread.

**Verified.** A prediction from a worker thread now matches one from the main thread. The regression test is `tests/test_prediction.py::test_make_prediction_works_from_another_thread`.

## 2. Invalid input crashed the API with 500

**Symptom.** `sub_grade: "A"`, `term: "48 months"`, `term: null`, a negative FICO score and `inputs: []` all returned `500 Internal Server Error`.

**Cause.**
- Every schema field was `Optional` with no constraints, so unknown categories reached the encoders, which raised `Found unknown categories`.
- `api.py` converted NaN to `None`, but `SimpleImputer` only treats `np.nan` as missing, so nulls crashed instead of being imputed.
- Unknown field names were silently ignored.
- The `/docs` example used the Pydantic v1 key `schema_extra`, which v2 ignores, and unrealistic values.

**Fix** (`src/processing/validation.py`, shared by the API and the package):
- `Literal` types for `term` and `sub_grade`, matching the trained encoders' categories.
- Ranges: FICO 300-850, rate 0-50, counts and percentages 0-100.
- Every field is required but nullable.
- `extra="forbid"`; batches of 1-1000 inputs.
- Whitespace and case are normalised.
- `None` is converted to `NaN` before the pipeline, so imputation works.
- A realistic `json_schema_extra` example for `/docs`.

**Verified.** Every case above now returns 422 with a per-field message, and null fields return 200 with imputed values. Covered by `tests/test_api.py` and `tests/test_features.py`.

## 3. Training script could not run

**Symptom.**
- `python src/train.py` raised `No module named 'src'`.
- `python -m src.train` raised `No module named 'pipeline'`.

**Cause.** `train.py` mixed `from pipeline ...` / `from config.core ...` imports with `from src.processing ...` imports. No single launch command satisfies both.

**Fix.**
- Consistent `src.`-prefixed imports. Run with `python -m src.train` from the repo root.
- Metrics are logged rather than printed.
- `load_dataset` was rewritten with `.isin()` and an explicit `.copy()`, which removes a `SettingWithCopyWarning`.

**Verified.** The full training run completed on a synthetic dataset with the same schema, using temporary folders so the real model was untouched.

## 4. "Default probability" was not a probability

**Symptom.** An A2 loan with a 750 FICO score and a 10.5% rate was shown as a 30% default probability.

**Cause.** `scale_pos_weight=3.66` (cost-sensitive learning for recall) shifts `predict_proba` upwards, so the output is a ranking score, not a calibrated probability.

**Fix.** The frontend and README now call it a **risk score** and explain why. Proper calibration (e.g. `CalibratedClassifierCV`, then re-choosing the decision threshold) needs a retrain on the full dataset. That's left as a follow-up modelling decision.

## 5. Tests could not catch bugs

**Cause.**
- `@patch("src.config.core.DATASET_DIR")` patched the definition, not the copy `loading.py` had already imported, so the patch did nothing.
- `isinstance(errors, Optional[dict])` and `isinstance(x, object)` are always true.
- There were no API tests.

**Fix.**
- Patch where names are used, and assert on the paths actually used.
- Parametrised validation tests.
- Prediction tests for error handling, imputation and threading.
- A new `tests/test_api.py`.
- A CI workflow (`.github/workflows/tests.yml`).

**Verified.** 5 tests became 33. Run against the original code, the new tests produce 12 errors and 1 failure.

## 6. Bloated runtime dependencies

**Cause.**
- One dependency list served the notebooks, training and the API, so the API image contained MLflow, CatBoost, LightGBM, Optuna, SHAP, Jupyter and plotting libraries.
- On Linux, the standard `xgboost` wheel also pulls a 294 MB NVIDIA GPU library (`nvidia-nccl-cu12`).
- `uvicorn` and `requests` were never declared; they arrived only as dependencies of MLflow and Streamlit.

**Fix.**
- Runtime dependencies are kept to what serving needs. Research libraries moved to a `training` dependency group (`uv sync --group training`), and `httpx` moved to `dev`.
- `xgboost-cpu` on Linux, pinned to 3.2.x like the saved model. Its predictions are identical.
- `uvicorn` and `requests` are declared explicitly.
- Both `uv.lock` and `poetry.lock` are regenerated, with other versions unchanged.

**Result.** Runtime downloads went from about 823 MB (148 packages) to about 165 MB (57 packages). A clean runtime install took 7 s, and the API was ready in 1.3 s.

## 7. Broken local Docker setup

**Cause.**
- `.dockerignore` did not exclude `.venv`, so `COPY . .` could overwrite the image's Linux virtualenv with a host one.
- `docker-compose.yml` bind-mounted `.:/app`, which hid the container's virtualenv, so `uvicorn` was not found.
- The frontend's default `localhost:8000` pointed at its own container.
- `uv run` at start-up re-synced the environment on every cold start.

**Fix.**
- Exclude `.venv`, `data/` and `notebooks/` from the image.
- A cached dependency layer, and run `uvicorn` / `streamlit` from the virtualenv.
- Compose keeps the container's `.venv`, sets `BACKEND_URL=http://api:8000/api/v1`, and waits for the API health check.

## 8. Demo slow or failing after idle periods

**Cause.**
- Render's free tier sleeps after 15 minutes idle and takes about a minute to wake.
- Streamlit Community Cloud sleeps after 12 hours without a browser visit.
- The frontend gave up after a hard 30 s timeout, or immediately on a refused connection.

**Fix.**
- `app.py` pings `/health` in the background when the page opens.
- `/predict` is retried for up to 2 minutes on connection errors, timeouts and 502/503/504, with a message explaining the wait.
- The API URL box is hidden when `BACKEND_URL` is configured.
- `.github/workflows/keep-streamlit-awake.yml` opens the app in headless Chromium every 6 hours and clicks the wake-up button if needed (`scripts/wake_streamlit.py`).
- `.github/workflows/keep-api-warm.yml` is a backup `/health` ping every 10 minutes. The primary pinger should be an external uptime monitor.

**Verified.**
- Cold start simulated with the API down for 20 s after the click: the app waited and showed the result with no error.
- The wake script was tested against a mock sleep page and the live app locally (asleep, awake and failure paths).

## 9. Debug prints and Pydantic v1 leftovers

**Fix.**
- Removed the config dump printed on every import, plus the debug prints in `features.py` and `predict.py`.
- `Settings` uses `SettingsConfigDict`, and `/health` uses `.model_dump()`.
- Predictions are returned as integers.
- `use_container_width` is replaced with `width="stretch"`.
