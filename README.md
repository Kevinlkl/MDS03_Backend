# MDS03 Backend

This repository is organized into two main folders:

- `backend/` — runnable FastAPI backend with inference endpoints
- `training_notebook/` — model training and analysis notebooks

## Run the backend

1. Create and activate your Python environment.
2. Install dependencies:

```powershell
cd backend
pip install -r requirements-sd.txt
```

3. Start the FastAPI server:

```powershell
python -m uvicorn main:app --reload
```

4. Open the API docs:

- http://127.0.0.1:8000/docs

## Backend folder structure

- `backend/main.py` — FastAPI application entrypoint
- `backend/api/` — inference endpoints
- `backend/model_t1_t2/` — T1-to-T2 model code
- `backend/model_t1_flair/` — T1-to-FLAIR model code
- `backend/model_t1_synthetic/` — synthetic T1 generation code
- `backend/model_classification/` — classification endpoint code

## Training notebooks

The `training_notebook/` folder contains Jupyter notebooks used for model training and experimentation.

## Notes

- The backend has been cleaned to include only runnable endpoint code.
- Unused helper files and unrelated training scripts were removed from the backend folder.
